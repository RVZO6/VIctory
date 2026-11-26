//! Turns a title's image into something the Windows loader maps but does
//! not run. Pure byte manipulation, so it is testable on any host.

use victory_pe::layout::{self, dir, dll_flags, file_flags};
use victory_pe::{ImageInfo, PeError};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot patch image: {0}")]
    Patch(&'static str),
}

/// A patched copy of an image plus everything we need from the original.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Parsed from the *original* bytes: imports, TLS callbacks and entry
    /// point as the title expects them.
    pub info: ImageInfo,
    /// The patched bytes to hand to the Windows loader.
    pub bytes: Vec<u8>,
    /// BLAKE3 of the original bytes: identifies the exact build.
    pub original_hash: blake3::Hash,
}

impl PreparedImage {
    /// File name for the patched copy, unique per original build.
    pub fn file_name(&self) -> String {
        format!("victory-image-{}.dll", &self.original_hash.to_hex()[..16])
    }
}

/// Produces the patched copy. The changes are:
///
/// * `IMAGE_FILE_DLL` set, entry point cleared: the OS maps it like a DLL
///   and calls nothing;
/// * import directory points to its empty terminator and the bound-import
///   directory is cleared: we bind the original slots using `info.imports`;
/// * TLS callback pointer cleared: the OS still allocates static TLS for
///   the image, but callbacks only run when we call them (after binding);
///   its base relocation is made a no-op so the 0 survives a rebase;
/// * `FORCE_INTEGRITY` cleared, checksum zeroed: the copy is unsigned.
pub fn prepare(original: &[u8]) -> Result<PreparedImage, PrepareError> {
    let info = ImageInfo::parse(original)?;
    let mut bytes = original.to_vec();

    let fh = layout::file_header_offset(&bytes).ok_or(PrepareError::Patch("no file header"))?;
    let opt =
        layout::optional_header_offset(&bytes).ok_or(PrepareError::Patch("no optional header"))?;

    let characteristics = info.characteristics | file_flags::DLL;
    write_u16(
        &mut bytes,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        characteristics,
    )?;
    write_u32(&mut bytes, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0)?;
    write_u32(&mut bytes, opt + layout::OPT_CHECKSUM, 0)?;
    let dll_characteristics = info.dll_characteristics & !dll_flags::FORCE_INTEGRITY;
    write_u16(
        &mut bytes,
        opt + layout::OPT_DLL_CHARACTERISTICS,
        dll_characteristics,
    )?;

    // Windows 10 assigns static TLS to the prepared DLL when its empty
    // import table is present. The parser has already walked one descriptor
    // per module up to the null terminator, regardless of the recorded Size.
    if let Some(at) = layout::data_directory_offset(&bytes, dir::IMPORT) {
        let rva = layout::read_u32(&bytes, at).ok_or(PrepareError::Patch("no import RVA"))?;
        if rva != 0 {
            let terminator = u32::try_from(info.imports.len())
                .ok()
                .and_then(|count| count.checked_mul(IMPORT_DESCRIPTOR_SIZE))
                .and_then(|offset| rva.checked_add(offset))
                .ok_or(PrepareError::Patch("import terminator RVA overflow"))?;
            write_u32(&mut bytes, at, terminator)?;
            write_u32(&mut bytes, at + 4, IMPORT_DESCRIPTOR_SIZE)?;
        }
    }
    if let Some(at) = layout::data_directory_offset(&bytes, dir::BOUND_IMPORT) {
        write(&mut bytes, at, &[0; layout::DATA_DIRECTORY_SIZE])?;
    }

    if let Some(tls) = &info.tls
        && tls.callbacks_va != 0
    {
        let at = info
            .rva_to_file_offset(tls.directory_rva)
            .ok_or(PrepareError::Patch("TLS directory is not file-backed"))?;
        write_u64(
            &mut bytes,
            at as usize + layout::TLS64_ADDRESS_OF_CALLBACKS,
            0,
        )?;
        // The field's base relocation would turn the 0 into `delta` whenever
        // the loader maps the copy away from its preferred base (Wine always
        // does for DYNAMIC_BASE images), and the loader then reads callbacks
        // from there. Make that entry a no-op.
        let field_rva = tls
            .directory_rva
            .checked_add(layout::TLS64_ADDRESS_OF_CALLBACKS as u32)
            .ok_or(PrepareError::Patch("TLS callback field RVA overflows"))?;
        clear_dir64_relocations(&mut bytes, &info, field_rva)?;
    }

    Ok(PreparedImage {
        info,
        bytes,
        original_hash: blake3::hash(original),
    })
}

/// Size of an IMAGE_IMPORT_DESCRIPTOR in the PE file.
const IMPORT_DESCRIPTOR_SIZE: u32 = 20;

/// Clears every `IMAGE_REL_BASED_DIR64` base relocation targeting `target_rva`
/// (turning it into `IMAGE_REL_BASED_ABSOLUTE`).
///
/// Each PE base-relocation block covers only `[page, page + 0xFFF]`, so blocks
/// outside `target_rva`'s page are skipped without decoding their entries or
/// allocating a `Vec` of every relocation in the image.
fn clear_dir64_relocations(
    bytes: &mut [u8],
    info: &ImageInfo,
    target_rva: u32,
) -> Result<(), PrepareError> {
    const DIR64: u16 = 10;
    let Some(d) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
        return Ok(());
    };
    let (Some(rva), Some(size)) = (layout::read_u32(bytes, d), layout::read_u32(bytes, d + 4))
    else {
        return Ok(());
    };
    let Some(start) = info.rva_to_file_offset(rva).filter(|_| size != 0) else {
        return Ok(());
    };
    let start = start as usize;
    let end = start.saturating_add(size as usize).min(bytes.len());
    let mut at = start;
    while at.saturating_add(8) <= end {
        let (Some(page), Some(block)) =
            (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
        else {
            break;
        };
        let block = block as usize;
        let Some(next) = at.checked_add(block) else {
            break;
        };
        if block < 8 || next > end {
            break;
        }
        if let Some(wanted_off) = target_rva.checked_sub(page).filter(|&d| d < 0x1000) {
            let wanted_entry = (DIR64 << 12) | (wanted_off as u16);
            let mut entry = at + 8;
            while entry + 2 <= next {
                let e = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
                if e == wanted_entry {
                    write_u16(bytes, entry, 0)?; // IMAGE_REL_BASED_ABSOLUTE
                }
                entry += 2;
            }
        }
        at = next;
    }
    Ok(())
}

fn write_u16(bytes: &mut [u8], at: usize, v: u16) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u32(bytes: &mut [u8], at: usize, v: u32) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u64(bytes: &mut [u8], at: usize, v: u64) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write(bytes: &mut [u8], at: usize, v: &[u8]) -> Result<(), PrepareError> {
    bytes
        .get_mut(at..at + v.len())
        .ok_or(PrepareError::Patch("header field outside of file"))?
        .copy_from_slice(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll(), "info describes the original");
        assert!(prepared.info.import_count() > 0);

        let patched = ImageInfo::parse(&prepared.bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        assert_eq!(patched.dll_characteristics & dll_flags::FORCE_INTEGRITY, 0);
        if let Some(tls) = &patched.tls {
            assert_eq!(tls.callbacks_va, 0);
            let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
            assert!(
                dir64_relocations(&prepared.bytes, &patched)
                    .iter()
                    .all(|&(_, target)| target != field),
                "the cleared callback pointer must not be relocated"
            );
        }
        // Nothing else moved: same sections, same size.
        assert_eq!(patched.sections.len(), prepared.info.sections.len());
        assert_eq!(prepared.bytes.len(), original.len());
    }

    #[test]
    fn keeps_iat_slots_when_import_size_excludes_the_terminator() {
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let info = ImageInfo::parse(&original).unwrap();
        let directory = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        // This linker layout counts the descriptors for real modules only.
        let descriptor_bytes = u32::try_from(info.imports.len()).unwrap() * IMPORT_DESCRIPTOR_SIZE;
        write_u32(&mut original, directory + 4, descriptor_bytes).unwrap();
        let prepared = prepare(&original).unwrap();
        let empty = layout::read_u32(&prepared.bytes, directory).unwrap();
        assert_eq!(
            empty,
            layout::read_u32(&original, directory).unwrap() + descriptor_bytes
        );
        assert_eq!(
            layout::read_u32(&prepared.bytes, directory + 4),
            Some(IMPORT_DESCRIPTOR_SIZE)
        );
        let at = info.rva_to_file_offset(empty).unwrap() as usize;
        assert_eq!(
            &prepared.bytes[at..at + IMPORT_DESCRIPTOR_SIZE as usize],
            &[0; 20]
        );
        let iat = layout::data_directory_offset(&original, dir::IAT).unwrap();
        assert_eq!(
            &prepared.bytes[iat..iat + layout::DATA_DIRECTORY_SIZE],
            &original[iat..iat + layout::DATA_DIRECTORY_SIZE]
        );
        assert_eq!(prepared.info.import_count(), info.import_count());
        for module in &prepared.info.imports {
            for entry in &module.entries {
                let at = info.rva_to_file_offset(entry.iat_slot_rva).unwrap() as usize;
                assert_eq!(&prepared.bytes[at..at + 8], &original[at..at + 8]);
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_assigns_static_tls_to_a_prepared_dll() {
        // Model a title DLL using our own linked image. Loading a DLL leaves
        // its assigned index intact; the executable-specific redirect resets it.
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let flags =
            layout::file_header_offset(&original).unwrap() + layout::FILE_HEADER_CHARACTERISTICS;
        let characteristics =
            ImageInfo::parse(&original).unwrap().characteristics | file_flags::DLL;
        write_u16(&mut original, flags, characteristics).unwrap();
        let prepared = prepare(&original).unwrap();
        let info = &prepared.info;
        let tls = info.tls.as_ref().expect("the test binary uses static TLS");
        let directory = info.rva_to_file_offset(tls.directory_rva).unwrap() as usize;
        let index_va = layout::read_u64(&original, directory + 16).unwrap();
        let index_rva = index_va.checked_sub(info.image_base).unwrap();
        assert!(index_rva + 4 <= u64::from(info.size_of_image));

        let folder = tempfile::tempdir().unwrap();
        let image = crate::load(&prepared, folder.path()).unwrap();
        // SAFETY: the mapped test image contains the checked DWORD above.
        let assigned =
            unsafe { ((image.base() + index_rva as usize) as *const u32).read_unaligned() };
        // SAFETY: the prepared DLL has never been attached or given to a caller.
        unsafe { image.unload() };
        assert_ne!(
            assigned, 0,
            "the process executable already owns TLS slot zero"
        );
    }

    #[test]
    fn file_name_is_stable_per_build() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let a = prepare(&original).unwrap();
        let b = prepare(&original).unwrap();
        assert_eq!(a.file_name(), b.file_name());
    }

    /// Reference full-table scan kept in tests to verify `clear_dir64_relocations`
    /// clears every matching entry and leaves every other entry untouched.
    fn dir64_relocations(bytes: &[u8], info: &ImageInfo) -> Vec<(usize, u32)> {
        const DIR64: u16 = 10;
        let mut out = Vec::new();
        let Some(d) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
            return out;
        };
        let (Some(rva), Some(size)) = (layout::read_u32(bytes, d), layout::read_u32(bytes, d + 4))
        else {
            return out;
        };
        let Some(start) = info.rva_to_file_offset(rva).filter(|_| size != 0) else {
            return out;
        };
        let (mut at, end) = (
            start as usize,
            (start as usize).saturating_add(size as usize).min(bytes.len()),
        );
        while at.saturating_add(8) <= end {
            let (Some(page), Some(block)) =
                (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
            else {
                break;
            };
            let block = block as usize;
            let Some(next) = at.checked_add(block) else {
                break;
            };
            if block < 8 || next > end {
                break;
            }
            let mut entry = at + 8;
            while entry + 2 <= next {
                let e = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
                if e >> 12 == DIR64
                    && let Some(target) = page.checked_add(u32::from(e & 0xFFF))
                {
                    out.push((entry, target));
                }
                entry += 2;
            }
            at = next;
        }
        out
    }

    /// Builds a minimal synthetic PE64 image whose `.reloc` section holds
    /// `blocks` base-relocation blocks of `entries_per_block` `DIR64` entries
    /// (plus one non-`DIR64` entry per block), starting at page `0x10000`.
    fn synthetic_reloc_pe(blocks: usize, entries_per_block: usize) -> (Vec<u8>, ImageInfo) {
        let block_bytes = 8 + (entries_per_block * 2).next_multiple_of(4);
        let reloc_size = blocks * block_bytes;
        let reloc_raw_size = reloc_size.next_multiple_of(0x200);
        let headers_size = 0x200usize;
        let mut pe = vec![0u8; headers_size + reloc_raw_size];

        // DOS header
        pe[0..2].copy_from_slice(b"MZ");
        pe[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());

        // PE signature + COFF header at 0x80 (24 bytes)
        let pe_off = 0x80usize;
        pe[pe_off..pe_off + 4].copy_from_slice(b"PE\0\0");
        let coff = pe_off + 4;
        pe[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes()); // AMD64
        pe[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // NumberOfSections = 1
        let opt_size: u16 = 0xF0;
        pe[coff + 16..coff + 18].copy_from_slice(&opt_size.to_le_bytes());
        pe[coff + 18..coff + 20].copy_from_slice(&0x0022u16.to_le_bytes()); // EXECUTABLE | LARGE_ADDRESS_AWARE

        // Optional header PE32+ at 0x98 (240 bytes)
        let opt = coff + 20;
        pe[opt..opt + 2].copy_from_slice(&0x020Bu16.to_le_bytes()); // PE32+
        pe[opt + 24..opt + 32].copy_from_slice(&0x1_4000_0000u64.to_le_bytes()); // ImageBase
        pe[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // SectionAlignment
        pe[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // FileAlignment
        let reloc_rva: u32 = 0x1000;
        let reloc_vsize = reloc_raw_size as u32;
        let size_of_image = reloc_rva + reloc_vsize.next_multiple_of(0x1000);
        pe[opt + 56..opt + 60].copy_from_slice(&size_of_image.to_le_bytes());
        pe[opt + 60..opt + 64].copy_from_slice(&(headers_size as u32).to_le_bytes());
        pe[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes

        // Data directory entry 5 (BASERELOC) at opt + 112 + 5 * 8
        let dd = opt + 112 + dir::BASERELOC * 8;
        pe[dd..dd + 4].copy_from_slice(&reloc_rva.to_le_bytes());
        pe[dd + 4..dd + 8].copy_from_slice(&(reloc_size as u32).to_le_bytes());

        // Single section header `.reloc` at opt + 0xF0 (40 bytes)
        let sec = opt + opt_size as usize;
        pe[sec..sec + 6].copy_from_slice(b".reloc");
        pe[sec + 8..sec + 12].copy_from_slice(&reloc_vsize.to_le_bytes()); // VirtualSize
        pe[sec + 12..sec + 16].copy_from_slice(&reloc_rva.to_le_bytes()); // VirtualAddress
        pe[sec + 16..sec + 20].copy_from_slice(&(reloc_raw_size as u32).to_le_bytes()); // SizeOfRawData
        pe[sec + 20..sec + 24].copy_from_slice(&(headers_size as u32).to_le_bytes()); // PointerToRawData
        pe[sec + 36..sec + 40].copy_from_slice(&0x4200_0040u32.to_le_bytes());

        // Populate base relocation blocks
        let mut cursor = headers_size;
        for b in 0..blocks {
            let page = 0x10000u32 + (b as u32) * 0x1000;
            pe[cursor..cursor + 4].copy_from_slice(&page.to_le_bytes());
            pe[cursor + 4..cursor + 8].copy_from_slice(&(block_bytes as u32).to_le_bytes());
            for i in 0..entries_per_block {
                let off = ((i * 8) & 0xFFF) as u16;
                // Make entry 0 a non-DIR64 type (HIGHLOW = 3) so we verify type filtering too.
                let ty: u16 = if i == 0 { 3 } else { 10 };
                let word = (ty << 12) | off;
                let p = cursor + 8 + i * 2;
                pe[p..p + 2].copy_from_slice(&word.to_le_bytes());
            }
            cursor += block_bytes;
        }

        let info = ImageInfo::parse(&pe).unwrap();
        (pe, info)
    }

    #[test]
    fn clear_dir64_relocations_matches_full_scan_and_preserves_neighbors() {
        let (orig, info) = synthetic_reloc_pe(64, 16);
        let targets = [
            0x10000 + 8,          // first block, DIR64
            0x10000,              // first block, non-DIR64 (type 3) -> must stay untouched
            0x10000 + 31 * 0x1000 + 48, // middle block, DIR64
            0x10000 + 63 * 0x1000 + 120, // last block, DIR64
            0x90000 + 16,         // absent page
        ];
        for &target in &targets {
            let mut fast = orig.clone();
            clear_dir64_relocations(&mut fast, &info, target).unwrap();

            let mut reference = orig.clone();
            for (entry, rva) in dir64_relocations(&orig, &info) {
                if rva == target {
                    write_u16(&mut reference, entry, 0).unwrap();
                }
            }
            assert_eq!(fast, reference, "mismatch for target {target:#x}");
        }
    }

    #[test]
    #[ignore]
    fn bench_clear_dir64_relocations() {
        use std::hint::black_box;
        use std::time::Instant;

        // 3,000 blocks x 250 entries = 750,000 relocations (retail-sized .reloc table).
        let (orig, info) = synthetic_reloc_pe(3_000, 250);
        let target = 0x10000 + 2_500 * 0x1000 + 80;
        let iters = 50usize;

        let mut buf_old = orig.clone();
        let t0 = Instant::now();
        let mut hits_old = 0usize;
        for _ in 0..iters {
            // Restore entry before each iteration so both paths do identical work.
            let matches = dir64_relocations(black_box(&buf_old), &info);
            for (entry, rva) in matches {
                if rva == target {
                    write_u16(&mut buf_old, entry, 0).unwrap();
                    hits_old += 1;
                }
            }
            let pos = 0x200 + 2_500 * (8 + (250 * 2usize).next_multiple_of(4)) + 8 + 10 * 2;
            write_u16(&mut buf_old, pos, (10u16 << 12) | 80).unwrap();
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut buf_new = orig.clone();
        let t1 = Instant::now();
        let mut hits_new = 0usize;
        for _ in 0..iters {
            clear_dir64_relocations(black_box(&mut buf_new), &info, target).unwrap();
            let pos = 0x200 + 2_500 * (8 + (250 * 2usize).next_multiple_of(4)) + 8 + 10 * 2;
            if u16::from_le_bytes([buf_new[pos], buf_new[pos + 1]]) == 0 {
                hits_new += 1;
            }
            write_u16(&mut buf_new, pos, (10u16 << 12) | 80).unwrap();
        }
        let new_ms = t1.elapsed().as_secs_f64() * 1e3;

        assert_eq!(hits_old, iters);
        assert_eq!(hits_new, iters);
        assert_eq!(buf_old, buf_new);
        eprintln!(
            "prepare_reloc[3000 blocks x 250 entries x{iters}]: old={old_ms:.2}ms new={new_ms:.4}ms speedup={:.1}x",
            old_ms / new_ms
        );
    }
}
