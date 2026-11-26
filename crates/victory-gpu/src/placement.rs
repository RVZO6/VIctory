//! Host textures placed in heaps made ahead instead of committed one by
//! one: a new shot brings hundreds of textures, and the host spent most of
//! a frame making them (300 textures, 430 MB on the RX 6700 XT: committed
//! 255 ms; one heap 32 ms, then placed 3 ms).
//!
//! Textures without render-target, depth or unordered-access use, up to a
//! quarter of a heap, go into heaps of [`HEAP`] bytes, first fit. A
//! texture's range is free again once nothing else holds its resource
//! (heap slots keep views' resources until the running submission has run,
//! lists until they are reset), so no view or list sees its memory reused.
//! Empty heaps past [`SPARE`] are released on the next reap (when existing
//! heaps fill during [`place`], on frame passes via [`free_bytes`]/[`stats`],
//! or once pending drops reach the protective [`REAP_THRESHOLD`]).

use std::collections::BTreeMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Bytes of one heap.
const HEAP: u64 = 64 << 20;
/// Empty heaps kept for the next textures.
const SPARE: usize = 2;
/// Placement alignment of textures that are not render targets.
const ALIGN: u64 = 64 << 10;
/// Minimum pending drops before `free()` runs a protective reap pass.
const REAP_THRESHOLD: usize = 1024;

/// Where a texture lies: heap, offset and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    heap: u64,
    offset: u64,
    size: u64,
}

/// Free byte ranges inside one heap, with tracked totals for O(1) skip of
/// full/fragmented heaps and O(1) empty checks.
struct FreeList {
    ranges: BTreeMap<u64, u64>,
    free_bytes: u64,
    max_free: u64,
}

impl FreeList {
    fn full(bytes: u64) -> Self {
        let mut ranges = BTreeMap::new();
        ranges.insert(0, bytes);
        Self {
            ranges,
            free_bytes: bytes,
            max_free: bytes,
        }
    }

    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        if self.max_free < size {
            return None;
        }
        let (offset, removed_len) = take_with_len(&mut self.ranges, size, align)?;
        self.free_bytes -= size;
        if removed_len == self.max_free {
            self.max_free = self.ranges.values().copied().max().unwrap_or(0);
        }
        Some(offset)
    }

    fn give(&mut self, offset: u64, size: u64) {
        self.free_bytes += size;
        let merged = give(&mut self.ranges, offset, size);
        if merged > self.max_free {
            self.max_free = merged;
        }
    }

    fn is_empty_heap(&self) -> bool {
        self.free_bytes == HEAP
    }
}

/// A heap and its free ranges (offset → bytes).
struct Heap {
    heap: ID3D12Heap,
    free: FreeList,
}

impl Heap {
    fn new(heap: ID3D12Heap) -> Self {
        Self {
            heap,
            free: FreeList::full(HEAP),
        }
    }

    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        self.free.take(size, align)
    }

    fn give(&mut self, offset: u64, size: u64) {
        self.free.give(offset, size);
    }

    fn is_empty(&self) -> bool {
        self.free.is_empty_heap()
    }
}

#[derive(Default)]
struct Heaps {
    heaps: BTreeMap<u64, Heap>,
    next: u64,
    /// Ranges whose texture may still be held elsewhere, with a holder of
    /// our own.
    pending: Vec<(ID3D12Resource, Placement)>,
    /// Protective reap high-water mark for drop-only loops (doubling backoff).
    next_reap_at: usize,
}

// SAFETY: D3D12 heaps and resources are free-threaded; used under the lock.
unsafe impl Send for Heaps {}

static HEAPS: Mutex<Option<Heaps>> = Mutex::new(None);

/// The first range of `free` that fits `size` bytes at `align`: its
/// offset and original range length, taken out of `free`.
fn take_with_len(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<(u64, u64)> {
    let (&start, &len, at) = free.iter().find_map(|(s, l)| {
        let at = s.next_multiple_of(align);
        (at + size <= s + l).then_some((s, l, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    Some((at, len))
}

#[cfg(test)]
fn take(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    take_with_len(free, size, align).map(|(at, _)| at)
}

/// Gives `[offset, offset + size)` back to `free`, joined with its
/// neighbours, returning the merged range length.
fn give(free: &mut BTreeMap<u64, u64>, offset: u64, size: u64) -> u64 {
    let (mut start, mut end) = (offset, offset + size);
    if let Some((&s, &l)) = free.range(..start).next_back()
        && s + l == start
    {
        free.remove(&s);
        start = s;
    }
    if let Some(&l) = free.get(&end) {
        free.remove(&end);
        end += l;
    }
    let merged = end - start;
    free.insert(start, merged);
    merged
}

/// Whether a resource has no holder but the one passed.
fn last_holder(resource: &ID3D12Resource) -> bool {
    // SAFETY: a live COM object; one reference taken and given back.
    unsafe {
        let raw = resource.as_raw();
        let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
        ((**unknown).AddRef)(raw);
        ((**unknown).Release)(raw) == 1
    }
}

impl Heaps {
    /// Frees the pending ranges nothing else holds now, and releases empty
    /// heaps past [`SPARE`].
    fn reap(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if last_holder(&self.pending[i].0) {
                let (_, p) = self.pending.swap_remove(i);
                if let Some(h) = self.heaps.get_mut(&p.heap) {
                    h.give(p.offset, p.size);
                }
            } else {
                i += 1;
            }
        }
        self.next_reap_at = self.pending.len().saturating_mul(2).max(REAP_THRESHOLD);
        let mut spare = 0;
        self.heaps.retain(|_, h| {
            if !h.is_empty() {
                return true;
            }
            spare += 1;
            spare <= SPARE
        });
    }

    fn allocate_existing(&mut self, size: u64, align: u64) -> Option<(ID3D12Heap, Placement)> {
        for (&id, h) in &mut self.heaps {
            if let Some(offset) = h.take(size, align) {
                return Some((h.heap.clone(), Placement { heap: id, offset, size }));
            }
        }
        None
    }

    /// A range of `size` bytes: in a heap with room, else after reaping
    /// pending releases, else in a new heap.
    fn allocate(&mut self, device: &ID3D12Device, size: u64, align: u64) -> Option<(ID3D12Heap, Placement)> {
        if let Some(found) = self.allocate_existing(size, align) {
            return Some(found);
        }
        if !self.pending.is_empty() {
            self.reap();
            if let Some(found) = self.allocate_existing(size, align) {
                return Some(found);
            }
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: HEAP,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: ALIGN,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation, our out pointer.
        unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
        let heap = heap?;
        crate::host::name(&heap, || "texture heap".into());
        let id = self.next;
        self.next += 1;
        let mut entry = Heap::new(heap.clone());
        let offset = entry.take(size, align)?;
        self.heaps.insert(id, entry);
        Some((heap, Placement { heap: id, offset, size }))
    }

    fn push_pending(&mut self, resource: ID3D12Resource, p: Placement) {
        self.pending.push((resource, p));
        let limit = self.next_reap_at.max(REAP_THRESHOLD);
        if self.pending.len() >= limit {
            self.reap();
        }
    }

    fn total_free_bytes(&self) -> u64 {
        self.heaps.values().map(|h| h.free.free_bytes).sum()
    }
}

/// Whether a texture of `desc` goes into a heap: no render-target, depth or
/// unordered-access use, and at most a quarter of a heap.
pub(crate) fn fits(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    const USES: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Flags.0 & USES != 0 || desc.SampleDesc.Count > 1 {
        return false;
    }
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    info.SizeInBytes != u64::MAX && info.SizeInBytes <= HEAP / 4 && info.Alignment <= ALIGN
}

/// Room for a texture of `desc` in a heap ([`fits`] first): the heap and
/// the placement, which the texture gives back with [`free`].
pub(crate) fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Heap, Placement)> {
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    let size = info.SizeInBytes.next_multiple_of(ALIGN);
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(Heaps::default);
    heaps.allocate(device, size, info.Alignment.max(ALIGN))
}

/// A range [`place`] gave whose texture could not be made there.
pub(crate) fn unplace(p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut().and_then(|hs| hs.heaps.get_mut(&p.heap)) {
        h.give(p.offset, p.size);
    }
}

/// The texture at `p` is let go of: its range is free again once nothing
/// else holds `resource` (checked when existing heaps fill, on frame passes
/// via [`free_bytes`]/[`stats`], or past the protective pending threshold).
pub(crate) fn free(resource: ID3D12Resource, p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut() {
        h.push_pending(resource, p);
    }
}

/// Bytes of heap space free for textures (after freeing what nothing
/// holds now): the card's memory they take is not in use.
pub(crate) fn free_bytes() -> u64 {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return 0 };
    h.reap();
    h.total_free_bytes()
}

/// Heaps of textures held, and their used and free bytes (after freeing
/// what nothing holds now), for the statistics line.
pub(crate) fn stats() -> (usize, u64, u64) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return (0, 0, 0) };
    h.reap();
    let free = h.total_free_bytes();
    (h.heaps.len(), h.heaps.len() as u64 * HEAP - free, free)
}

impl Placement {
    /// Its offset in its heap.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture's range is not handed out again while anything else holds
    /// its resource (a view's slot, a list), and is once nothing does.
    #[test]
    fn a_range_is_reused_only_once_nothing_holds_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 9,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(fits(&host.device, &desc));
        let mut heaps = Heaps::default();
        let (heap, a) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a range of the heap, our out pointer.
        unsafe { host.device.CreatePlacedResource(&heap, a.offset(), &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture) }.unwrap();
        let texture = texture.unwrap();
        let held = texture.clone();
        heaps.pending.push((texture, a));
        heaps.reap();
        let (_, b) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_ne!(b, a, "still held: not handed out");
        drop(held);
        heaps.reap();
        let (_, c) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_eq!(c, a, "nothing holds it: free again");
    }

    #[test]
    fn ranges_are_taken_first_fit_and_joined_when_given_back() {
        let k = 64 << 10;
        let mut free = BTreeMap::from([(0, 10 * k)]);
        let a = take(&mut free, 3 * k, k).unwrap();
        let b = take(&mut free, 2 * k, k).unwrap();
        let c = take(&mut free, 4 * k, k).unwrap();
        assert_eq!((a, b, c), (0, 3 * k, 5 * k));
        assert_eq!(take(&mut free, 2 * k, k), None, "one 64 KiB range left");
        give(&mut free, b, 2 * k);
        assert_eq!(take(&mut free, 3 * k, k), None, "two ranges of 2 and 1, not joined");
        give(&mut free, c, 4 * k);
        assert_eq!(free, BTreeMap::from([(3 * k, 7 * k)]), "joined with both neighbours");
        give(&mut free, a, 3 * k);
        assert_eq!(free, BTreeMap::from([(0, 10 * k)]));
        // An aligned range inside a free one keeps the padding before it.
        let mut free = BTreeMap::from([(k, 10 * k)]);
        assert_eq!(take(&mut free, k, 4 * k), Some(4 * k));
        assert_eq!(free, BTreeMap::from([(k, 3 * k), (5 * k, 6 * k)]));
    }

    /// Pending drops do not reap on every `free()`, `allocate()` reaps on
    /// demand when a heap fills rather than allocating a new heap, and
    /// `free_bytes`/`max_free` stay exact across splits and merges.
    #[test]
    fn deferred_free_and_on_demand_reap_reuse_without_extra_heaps() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(fits(&host.device, &desc));
        let mut heaps = Heaps::default();
        // Fill one entire 64 MiB heap with 4 x 16 MiB placements.
        let chunk = HEAP / 4;
        let mut placed = Vec::new();
        for _ in 0..4 {
            let (heap, p) = heaps.allocate(&host.device, chunk, ALIGN).unwrap();
            let mut tex: Option<ID3D12Resource> = None;
            // SAFETY: placed inside heap at aligned offset.
            unsafe {
                host.device.CreatePlacedResource(
                    &heap,
                    p.offset(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut tex,
                )
            }
            .unwrap();
            placed.push((tex.unwrap(), p));
        }
        assert_eq!(heaps.heaps.len(), 1);
        assert_eq!(heaps.total_free_bytes(), 0);
        assert_eq!(heaps.heaps[&0].free.max_free, 0);

        // Push all 4 to pending with no external holder: deferred until demand or reap.
        for (tex, p) in placed {
            heaps.push_pending(tex, p);
        }
        assert_eq!(heaps.pending.len(), 4);

        // Next allocate() finds the existing heap full, reaps pending on demand,
        // and reuses offset 0 in heap 0 without allocating a second 64 MiB heap!
        let (_, reused) = heaps.allocate(&host.device, chunk, ALIGN).unwrap();
        assert_eq!(heaps.heaps.len(), 1);
        assert_eq!(reused.heap, 0);
        assert_eq!(reused.offset, 0);
        assert_eq!(heaps.total_free_bytes(), HEAP - chunk);
        assert_eq!(heaps.heaps[&0].free.max_free, HEAP - chunk);
        assert!(heaps.pending.is_empty());
    }

    #[test]
    fn free_list_metadata_stays_exact_across_fragmentation_and_coalescing() {
        let mut fl = FreeList::full(HEAP);
        assert!(fl.is_empty_heap());
        // Allocate 16 x 4 MiB blocks to fill the heap completely.
        let unit = 4 << 20;
        let mut offs = Vec::new();
        for _ in 0..16 {
            let o = fl.take(unit, ALIGN).unwrap();
            offs.push(o);
        }
        assert_eq!(fl.free_bytes, 0);
        assert_eq!(fl.max_free, 0);
        assert_eq!(fl.take(ALIGN, ALIGN), None);

        // Free even blocks -> 8 disjoint 4 MiB holes.
        for &o in offs.iter().step_by(2) {
            fl.give(o, unit);
        }
        assert_eq!(fl.free_bytes, 8 * unit);
        assert_eq!(fl.max_free, unit);
        assert_eq!(fl.take(unit * 2, ALIGN), None);

        // Free block 1 -> merges 0 + 1 + 2 = 12 MiB!
        fl.give(offs[1], unit);
        assert_eq!(fl.free_bytes, 9 * unit);
        assert_eq!(fl.max_free, 3 * unit);
        let big = fl.take(3 * unit, ALIGN).unwrap();
        assert_eq!(big, 0);
        assert_eq!(fl.max_free, unit);
        fl.give(big, 3 * unit);

        // Free remaining odd blocks -> coalesces back to full HEAP.
        for &o in offs.iter().skip(3).step_by(2) {
            fl.give(o, unit);
        }
        assert!(fl.is_empty_heap());
        assert_eq!(fl.free_bytes, HEAP);
        assert_eq!(fl.max_free, HEAP);
        assert_eq!(fl.ranges.len(), 1);
    }
}
