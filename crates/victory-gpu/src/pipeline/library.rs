//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = read(LIBRARY_FILE).unwrap_or_default();
            let tag = tag(&blob);
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(PREBUILT_FILE))
                && !l._blob.is_empty()
            {
                l.prebuilt = listed(&list, &tag);
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        let started = std::time::Instant::now();
        match make(&blob) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                tracing::info!(bytes = blob.len(), ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    name_with(kind, root, code, |out| out.write_str(state))
}

/// A pipeline's name in the library, streaming its state description into the
/// hasher with no intermediate `String` allocation while keeping the exact
/// length-prefixed hash layout of [`name`].
pub(crate) fn name_with(
    kind: &str,
    root: &[u8],
    code: &[&[u8]],
    write_state: impl Fn(&mut dyn std::fmt::Write) -> std::fmt::Result,
) -> String {
    struct Len(u64);
    impl std::fmt::Write for Len {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0 += s.len() as u64;
            Ok(())
        }
    }

    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    let mut len = Len(0);
    let _ = write_state(&mut len);
    h.update(&len.0.to_le_bytes());
    let _ = write_state(&mut super::Blake3Writer(&mut h));
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let _held = NAMES.hold(name);
    // The interface alone under the lock: loads take up to seconds. Its
    // blob lives as long as the library, which is never dropped.
    let loads = library
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .library
        .clone();
    if let Ok(pso) = load(&loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let (blob, prebuilt) = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !l.prebuilt_changed && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                l.prebuilt_changed = false;
                // Kept beside the bytes it holds for (also when a run adds
                // pipelines).
                let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&bytes), &l.prebuilt));
                (bytes, prebuilt)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used = EARLY_BEFORE
        .get()
        .map(|before| {
            let now = USED_NOW.lock().unwrap_or_else(|e| e.into_inner());
            match now.as_ref() {
                Some(now) => early_list(before, now),
                None => early_list(before, &KeySet::default()),
            }
        })
        .unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::AtomicU64;
    static WRITING: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    let tmp = path.with_file_name(name);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Parses a canonical 32-character lowercase hexadecimal pipeline key as `u128`.
/// Non-canonical keys (different length, uppercase, sign, non-hex) return `None`
/// and fall back to exact string storage so semantics and round-tripping stay identical.
fn parse_key_u128(key: &str) -> Option<u128> {
    (key.len() == 32 && key.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then(|| u128::from_str_radix(key, 16).ok())
        .flatten()
}

/// Compact set of pipeline keys: canonical 32-hex keys live in `HashSet<u128>`
/// with zero per-key heap `String` allocations; any non-32-hex line falls back
/// to `other`.
#[derive(Default)]
struct KeySet {
    hex: HashSet<u128>,
    other: Vec<String>,
}

impl KeySet {
    fn from_lines(text: &str) -> Self {
        let mut set = Self::default();
        for line in text.lines().filter(|l| !l.is_empty()) {
            set.insert(line);
        }
        set
    }

    fn insert(&mut self, key: &str) -> bool {
        if let Some(id) = parse_key_u128(key) {
            self.hex.insert(id)
        } else if self.other.iter().any(|s| s == key) {
            false
        } else {
            self.other.push(key.to_owned());
            true
        }
    }

    fn contains(&self, key: &str) -> bool {
        if let Some(id) = parse_key_u128(key) {
            self.hex.contains(&id)
        } else {
            self.other.iter().any(|s| s == key)
        }
    }

    fn len(&self) -> usize {
        self.hex.len() + self.other.len()
    }
}

/// Title pipelines created, and their distinct keys.
#[derive(Default)]
struct Created {
    total: u64,
    distinct: KeySet,
}

impl Created {
    fn record(&mut self, key: &str) {
        self.total += 1;
        self.distinct.insert(key);
    }

    fn counts(&self) -> (u64, usize) {
        (self.total, self.distinct.len())
    }
}

static CREATED: Mutex<Option<Created>> = Mutex::new(None);

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.get_or_insert_with(Created::default).record(key);
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.as_ref().map_or((0, 0), Created::counts)
}

/// The list to keep: this run's keys, then earlier runs' others, at most
/// `EARLY_MAX` (the oldest go once it is full).
fn early_list(before: &KeySet, now: &KeySet) -> Vec<String> {
    let mut out = Vec::with_capacity((now.len() + before.len()).min(EARLY_MAX));
    for &id in &now.hex {
        if out.len() == EARLY_MAX {
            return out;
        }
        out.push(format!("{id:032x}"));
    }
    for key in &now.other {
        if out.len() == EARLY_MAX {
            return out;
        }
        out.push(key.clone());
    }
    for &id in &before.hex {
        if out.len() == EARLY_MAX {
            return out;
        }
        if !now.hex.contains(&id) {
            out.push(format!("{id:032x}"));
        }
    }
    for key in &before.other {
        if out.len() == EARLY_MAX {
            return out;
        }
        if !now.other.iter().any(|k| k == key) {
            out.push(key.clone());
        }
    }
    out
}

/// Earlier runs' early-start keys, loaded once so [`wanted_early`] is lock-free.
static EARLY_BEFORE: OnceLock<KeySet> = OnceLock::new();
/// Title pipeline keys drawn or dispatched with in this run.
static USED_NOW: Mutex<Option<KeySet>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn load_early_before(dir: Option<&std::path::Path>) -> KeySet {
    let listed = dir
        .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
        .unwrap_or_default();
    let before = KeySet::from_lines(&listed);
    tracing::info!(pipelines = before.len(), "title pipelines to start early");
    before
}

fn early_before() -> &'static KeySet {
    EARLY_BEFORE.get_or_init(|| load_early_before(DIR.get().map(PathBuf::as_path)))
}

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
pub(crate) fn keep_early_list() {
    let _ = early_before();
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    early_before().contains(key)
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let _ = early_before();
    let mut now = USED_NOW.lock().unwrap_or_else(|e| e.into_inner());
    if now.get_or_insert_with(KeySet::default).insert(key) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| {
            let mut set = KeySet::default();
            for i in r {
                set.insert(&format!("{i:032x}"));
            }
            set
        };
        let before = keys(0..EARLY_MAX);
        let now = keys(EARLY_MAX - 10..EARLY_MAX + 10);
        let list = early_list(&before, &now);
        assert_eq!(list.len(), EARLY_MAX);
        assert!((EARLY_MAX - 10..EARLY_MAX + 10).all(|i| list.contains(&format!("{i:032x}"))));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Vec::new()).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        let again = open(&device, made.serialized()).expect("the library again");
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn parse_key_u128_accepts_only_canonical_32_hex() {
        assert_eq!(
            parse_key_u128("00000000000000000000000000000000"),
            Some(0)
        );
        assert_eq!(
            parse_key_u128("ffffffffffffffffffffffffffffffff"),
            Some(u128::MAX)
        );
        assert_eq!(
            parse_key_u128("0123456789abcdef0123456789abcdef"),
            Some(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef)
        );
        for non_canonical in [
            "",
            "deadbeef",
            "0123456789abcdef0123456789abcde",   // 31 chars
            "0123456789abcdef0123456789abcdef0", // 33 chars
            "0123456789ABCDEF0123456789ABCDEF",  // uppercase
            "+0123456789abcdef0123456789abcde",  // leading +
            "-0123456789abcdef0123456789abcde",  // leading -
            " 0123456789abcdef0123456789abcde",  // leading space
            "0123456789abcdef0123456789abcdeg",  // non-hex char
        ] {
            assert_eq!(parse_key_u128(non_canonical), None, "{non_canonical:?}");
        }
    }

    #[test]
    fn created_deduplicates_hex_and_non_hex_keys() {
        let mut created = Created::default();
        assert_eq!(created.counts(), (0, 0));

        let k1 = "0123456789abcdef0123456789abcdef";
        let k2 = "fedcba9876543210fedcba9876543210";
        let upper = "0123456789ABCDEF0123456789ABCDEF";
        let plus = "+0000000000000000000000000000001";
        let custom = "legacy-non-hex-key";

        for k in [k1, k1, k2, upper, upper, plus, custom, custom, k2] {
            created.record(k);
        }
        assert_eq!(created.counts(), (9, 5));
        assert_eq!(created.distinct.hex.len(), 2);
        assert_eq!(created.distinct.other.len(), 3);
    }

    #[test]
    fn early_before_lock_free_lookup_handles_hex_and_non_hex() {
        let dir = tempfile::tempdir().unwrap();
        let hex_a = "00000000000000000000000000000001";
        let hex_b = "abcdef0123456789abcdef0123456789";
        let hex_missing = "abcdef0123456789abcdef0123456780";
        let upper = "ABCDEF0123456789ABCDEF0123456789";
        let plus = "+0000000000000000000000000000001";
        let non_hex = "custom-pipeline-key";

        let content = format!(
            "\n{hex_a}\n{hex_b}\n{hex_a}\n\n{upper}\n{plus}\n{non_hex}\n{non_hex}\n"
        );
        std::fs::write(dir.path().join(EARLY_FILE), content).unwrap();

        let cell: OnceLock<KeySet> = OnceLock::new();
        let loaded = cell.get_or_init(|| load_early_before(Some(dir.path())));
        assert_eq!(loaded.hex.len(), 2);
        assert_eq!(loaded.other.len(), 3);
        assert_eq!(loaded.len(), 5);

        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    let before = cell.get().unwrap();
                    assert!(before.contains(hex_a));
                    assert!(before.contains(hex_b));
                    assert!(!before.contains(hex_missing));
                    assert!(before.contains(upper));
                    assert!(before.contains(plus));
                    assert!(before.contains(non_hex));
                    assert!(!before.contains("00000000000000000000000000000002"));
                    assert!(!before.contains("other-missing"));
                });
            }
        });

        // Missing directory / file loads cleanly as empty.
        assert_eq!(load_early_before(None).len(), 0);
        let empty_dir = tempfile::tempdir().unwrap();
        assert_eq!(load_early_before(Some(empty_dir.path())).len(), 0);

        // `early_list` merges both hex and fallback keys with `now` first and no duplicates.
        let mut now = KeySet::default();
        now.insert(hex_b);
        now.insert(hex_missing);
        now.insert(non_hex);
        now.insert("new-custom");
        let merged = early_list(loaded, &now);
        assert_eq!(merged.len(), 7);
        assert_eq!(merged.iter().collect::<HashSet<_>>().len(), 7);
        for expected in [hex_a, hex_b, hex_missing, upper, plus, non_hex, "new-custom"] {
            assert!(merged.iter().any(|k| k == expected), "missing {expected}");
        }
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }
}
