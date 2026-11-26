use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use crate::ir::{Program, Value};

/// Fast non-cryptographic hasher for internal lowering lookup maps.
#[derive(Default)]
pub(super) struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(i)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(i)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(super) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub(super) type FxSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;

/// Dense table keyed by [`Value`], avoiding per-operand hashing in [`super::Lower`].
pub(super) struct DenseMap<T> {
    slots: Vec<Option<T>>,
}

impl<T: Copy> DenseMap<T> {
    pub(super) fn with_capacity(len: usize) -> Self {
        Self {
            slots: vec![None; len],
        }
    }

    #[inline]
    pub(super) fn get(&self, v: &Value) -> Option<&T> {
        self.slots.get(v.0 as usize).and_then(Option::as_ref)
    }

    #[inline]
    pub(super) fn contains_key(&self, v: &Value) -> bool {
        self.slots
            .get(v.0 as usize)
            .is_some_and(Option::is_some)
    }

    #[inline]
    pub(super) fn insert(&mut self, v: Value, val: T) {
        let i = v.0 as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = Some(val);
    }

    #[inline]
    pub(super) fn entry(&mut self, v: Value) -> DenseEntry<'_, T> {
        let i = v.0 as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        DenseEntry(&mut self.slots[i])
    }
}

pub(super) struct DenseEntry<'a, T>(&'a mut Option<T>);

impl<'a, T> DenseEntry<'a, T> {
    #[inline]
    pub(super) fn or_insert(self, default: T) -> &'a mut T {
        self.0.get_or_insert(default)
    }
}

/// One past the highest [`Value`] id in `p`.
pub(super) fn value_count(p: &Program) -> usize {
    let mut max = 0usize;
    for &(_, v) in &p.inputs {
        max = max.max(v.0 as usize + 1);
    }
    for block in &p.blocks {
        for phi in &block.phis {
            max = max.max(phi.value.0 as usize + 1);
        }
        for inst in &block.insts {
            for &(_, v) in &inst.writes {
                max = max.max(v.0 as usize + 1);
            }
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::lower::{Parts, lower, round16, round16_after};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn dense_map_and_round16_fast_path_behave_identically() {
        let mut m = DenseMap::<u32>::with_capacity(4);
        assert_eq!(m.get(&Value(0)), None);
        assert!(!m.contains_key(&Value(99)));
        m.insert(Value(2), 42);
        m.insert(Value(10), 7);
        assert_eq!(m.get(&Value(2)), Some(&42));
        assert_eq!(m.get(&Value(10)), Some(&7));
        assert!(m.contains_key(&Value(10)));
        assert!(!m.contains_key(&Value(3)));

        // Shader with no s_round_mode / s_setreg*: fast path gives Some(0) for every block.
        let plain: Vec<u8> = [0x0600_0100u32, S_ENDPGM]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p = ir::build(&plain, false, 8).unwrap();
        assert_eq!(round16(&p), vec![Some(0)]);

        // Shader with s_round_mode 0xC (bits 3:2 = 3, RTZ): full dataflow path still tracks it.
        let rtz: Vec<u8> = [
            0xBFA4_000Cu32, // s_round_mode 0xC (RTZ for f16/f64)
            0xBF82_0000,    // s_branch +0 -> next block
            0x0600_0100,
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p2 = ir::build(&rtz, false, 8).unwrap();
        let modes = round16(&p2);
        assert_eq!(modes[0], Some(0));
        assert_eq!(round16_after(&p2.blocks[0].insts[0], Some(0)), Some(3));
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_lower_hot_paths() {
        // Large 32-bit compute shader (2,500 VALU + 256 buffer loads + 16 branches)
        // exercising Lower::get, valu16 fast-reject, Lower::inst, Lower::buffer, and Lower::run.
        let mut words = vec![
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        ];
        for i in 0..256u32 {
            words.extend_from_slice(&[
                0xE030_2000,
                0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen
                0x0602_0502, // v_add_f32 v1, v2, v2
                0x1002_0301, // v_mul_f32 v1, v1, v1
                0x4A00_0101, // v_add_nc_u32 v0, v1, v0
            ]);
            for j in 0..7u32 {
                let dst = 3 + ((i + j) & 3);
                words.push(0x0600_0301 | (dst << 17)); // v_add_f32 v{dst}, v1, v1
            }
        }
        words.extend_from_slice(&[
            0xE070_2000,
            0x8002_0000, // buffer_store_dword v0, v0, s[8:11], 0 idxen
            S_ENDPGM,
        ]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 16,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, false, 16).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
        };
        // Warmup + FNV-1a checksum to verify bit-exact DXIL before/after.
        let out = lower(&parts).unwrap();
        let hash = out
            .dxil
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
            });

        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            let l = lower(&parts).unwrap();
            std::hint::black_box(l.dxil.len());
        }
        let lower_us = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
        eprintln!(
            "lower_hot ({} insts): {lower_us:.1} us/run, dxil={} B, fnv1a={hash:#018x}",
            p.blocks[0].insts.len(),
            out.dxil.len()
        );
    }
}
