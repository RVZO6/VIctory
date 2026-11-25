//! End-to-end tests of descriptors whose emission needs more than one
//! binding: a buffer read by element through the resource heap (no
//! used-resource entry, so no stride; the view-stride table gives it), and
//! one of two immediate samplers chosen by uniform control flow. Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP.

use windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC;

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_UINT,
};

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, StaticSampler, VIEW_INTEGER, VIEW_STRIDES, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 the heap index of a structured buffer (a root constant),
/// s1 the resource heap, s2 a table with a raw UAV. Each lane stores dword
/// 1 of element `lane` to dword `lane` of the UAV.
const HEAP_ELEMENT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_2004,
    0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a 2x1 texture, s1 a root constant, s2 a table
/// with a raw UAV. The root constant picks the immediate sampler by a
/// branch, clamping (non-zero) or wrapping (zero); the program samples the
/// texture at u = 1.25 and stores red to dword 0 of the UAV.
const CHOSEN_SAMPLER_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0006, // s_cbranch_scc1 -> 0x40
    0xBE94_03FF,
    0x0000_0092, // s_mov_b32 s20, 0x92 (clamp x, y, z)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch -> 0x50
    0xBE94_0380, // 0x40: s_mov_b32 s20, 0 (wrap)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0x7E00_02FF,
    0x3FA0_0000, // 0x50: v_mov_b32 v0, 1.25
    0x7E02_02FF,
    0x3F00_0000, // v_mov_b32 v1, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0200, // image_sample_lz v2, v[0:1], s[8:15], s[20:23] dmask:0x1 dim:2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_0280, // v_mov_b32 v3, 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 the heap index of a constant buffer (a root constant), s1
/// the resource heap, s2 a table with a raw UAV. Each lane stores dword 2
/// of the constant buffer (a scalar buffer load) to dword `lane` of the UAV.
const HEAP_CONSTANT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0008, // s_buffer_load_dword s20, s[8:11], 0x8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0214, // v_mov_b32 v1, s20
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xE070_1000,
    0x8004_0102, // buffer_store_dword v1, v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a raw SRV, s2 a table with a raw UAV. Each
/// lane loads the half at byte `2 × lane` and stores it to dword `lane`.
const HALF_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE028_1000,
    0x8002_0201, // buffer_load_ushort v2, v1, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Each lane stores its index as the
/// half at byte `2 × lane` (two lanes share each dword).
const HALF_STORE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0001, // buffer_store_short v0, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

/// An entry at offset 0 of the table in user data `user_data`.
fn entry(kind: ResourceType, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    }
}

fn build(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

#[test]
fn a_heap_buffer_read_by_element_takes_the_views_stride() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    // 32 elements of 12 bytes: (100 + e, 200 + e, 300 + e).
    let data: Vec<u32> = (0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]).collect();
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Structured(12),
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The buffer is at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds
                    // 12-byte elements.
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(|l| 200 + l).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A buffer read at a heap index from data whose slot holds no buffer view
/// (the view-stride table holds 0 there; a texture here) reads the host's
/// null buffer view instead (`emit`, "Buffer views at computed heap
/// indices"): a texture descriptor read as a buffer one is a wild address.
#[test]
fn a_heap_buffer_read_where_no_buffer_view_is_reads_zeros() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let texel = [[1.0f32, 2.0, 3.0, 4.0]];
    let views = [View::RawUav(32), View::Texture(1, 1, &texel), View::NullBuffers];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The texture at heap index 1, the UAV's table at 0.
            ud[u.index as usize] = u32::from(u.root == RootType::Constants);
        }
        let dispatch = Dispatch {
            // Heap index 1 holds no buffer view: stride 0.
            roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 0] })],
            views: &views,
            samplers: &[],
            groups: [1, 1, 1],
        };
        let words = host
            .dispatch(&root, &pso, &dispatch)
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], [0; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Constant data through the resource heap itself is read from a raw view
/// (the host's constant buffer views need an alignment the title's lack).
#[test]
fn a_constant_buffer_through_the_heap_reads_as_a_raw_buffer() {
    let shader = build(
        HEAP_CONSTANT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert!(shader.heap && !shader.heap_writes, "reads the heap itself, writes through a table");
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the guard's stride table");
    let data: Vec<u32> = vec![7, 8, 9, 10];
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The constants are at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds a
                    // buffer view.
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 4] })],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], vec![9; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_load_ushort`: the half at its byte address, zero-extended (the
/// BVH builder reads 16-bit fields this way).
#[test]
fn a_half_buffer_load_reads_the_half_at_its_address() {
    let shader = build(
        HALF_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    // 32 halves 0x8000 + 3 × lane, two per dword (low half first).
    let half = |l: u32| 0x8000 + 3 * l;
    let data: Vec<u32> = (0..16).map(|k| half(2 * k) | half(2 * k + 1) << 16).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 0, the UAV's at 1.
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(half).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_store_short`: each lane's half lands at its byte address and
/// leaves the other half of the dword alone.
#[test]
fn half_buffer_stores_leave_the_other_half_alone() {
    let shader = build(
        HALF_STORE_WAVE32,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let views = [View::RawUav(16)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..16).map(|k| (2 * k) | ((2 * k + 1) << 16)).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn a_sampler_chosen_by_a_uniform_branch_samples_as_chosen() {
    let shader = build(
        CHOSEN_SAMPLER_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Srv, 8, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
    );
    assert_eq!(shader.samplers.len(), 2);
    // Red 1 at u < 0.5, 2 from there: u = 1.25 wraps to 0.25, clamps to 1.
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let views = [View::Texture(2, 1, &texels), View::RawUav(1)];
    let samplers: [D3D12_SAMPLER_DESC; 0] = [];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (clamp, want) in [(1, 2.0f32), (0, 1.0)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Constants, 1) => clamp,
                    // The UAV at heap index 1: 32 bytes in.
                    (RootType::Table, 2) => 32,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(f32::from_bits(words[0][0]), want, "{label}, clamp {clamp}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
/// User data: s0 a table with a typed buffer UAV, s1 a table with a raw
/// buffer of 2 dwords a lane, s2 a table with a raw UAV. Each lane stores
/// its 4 halves to element `lane` of the typed view (`_d16`), loads them
/// back the same way and stores what it read to dwords `2 lane, 2 lane + 1`
/// of the raw UAV.
const D16_ROUND_TRIP_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0x0
    0xBE94_0302, // s_mov_b32 s20, s2
    0xBE95_0380, // s_mov_b32 s21, 0
    0xF408_040A,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
    0x3406_0083, // v_lshlrev_b32 v3, 3, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE034_1000,
    0x8003_0103, // buffer_load_dwordx2 v[1:2], v3, s[12:15], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE21C_2000,
    0x8002_0100, // buffer_store_format_d16_xyzw v[1:2], v0, s[8:11], 0 idxen
    0xE20C_2000,
    0x8002_0400, // buffer_load_format_d16_xyzw v[4:5], v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE074_1000,
    0x8004_0403, // buffer_store_dwordx2 v[4:5], v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a typed buffer SRV, s1 a table with a raw UAV.
/// Each lane loads element `lane` (`_d16_x`) and stores the dword it got to
/// dword `lane` of the raw UAV.
const D16_X_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0x7E08_0280, // v_mov_b32 v4, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE200_2000,
    0x8002_0400, // buffer_load_format_d16_x v4, v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0403, // buffer_store_dword v4, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Runs `shader` (user data: tables at heap indices 0, 1, 2 in order) on
/// hardware and WARP with `views` and the view-stride table `strides`;
/// each adapter's UAV contents.
fn run_tables(shader: &HostShader, views: &[View], strides: Vec<u32>) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // Table k is the view at heap index k (32 bytes a view).
            ud[u.index as usize] = 32 * u32::from(u.slot);
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: strides.clone() })],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words));
    }
    assert!(!out.is_empty(), "neither hardware nor WARP SM 6.6 device available");
    out
}

/// 16-bit typed stores and loads (`_d16`) through views of integer formats
/// pass the bits (the console's buffer format decides the conversion; d16
/// only packs 16-bit results), and through float views convert as halves:
/// Grand Theft Auto VI's face network keeps its halves in UINT views.
#[test]
fn d16_typed_buffer_accesses_pass_bits_through_integer_views() {
    let shader = build(
        D16_ROUND_TRIP_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Table, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Uav, 4, 0),
            entry(ResourceType::Srv, 4, 1),
            entry(ResourceType::Uav, 4, 2),
        ],
        32,
    );
    // Lane l: negative zero and small, signalling NaN, denormal and
    // all-ones patterns as bits; normal halves for the float view.
    let bits: Vec<u32> = (0..32u32)
        .flat_map(|l| [(0x8000 | l) | (0x7C01 + l) << 16, (1 + l) | (0xFFFF - l) << 16])
        .collect();
    let halves: Vec<u32> = (0..32u32)
        .flat_map(|l| [(0x3C00 + l) | (0xC000 + 3 * l) << 16, (0x1400 + l) | (0x5BFF - l) << 16])
        .collect();
    let cases: [(DXGI_FORMAT, u32, &[u32]); 2] = [
        (DXGI_FORMAT_R16G16B16A16_UINT, 8 | VIEW_INTEGER, &bits),
        (DXGI_FORMAT_R16G16B16A16_FLOAT, 8, &halves),
    ];
    for (format, stride, data) in cases {
        let views = [
            View::TypedUav(format, 32, 8),
            View::Buffer { data, stride: Stride::Raw },
            View::RawUav(64),
        ];
        for (label, words) in run_tables(&shader, &views, vec![stride, 4, 4]) {
            assert_eq!(words[0], data, "{label} {format:?}: the typed view's bytes");
            assert_eq!(words[1], data, "{label} {format:?}: what the loads read");
        }
    }
}

/// A one-channel `_d16_x` load from a typed SRV: an integer view's 16 bits
/// as they are, a float view's half as it is.
#[test]
fn d16_x_loads_read_r16_uint_and_r16_float_views() {
    let shader = build(
        D16_X_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 1)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 1)],
        32,
    );
    let pack = |h: &dyn Fn(u32) -> u32| -> Vec<u32> { (0..16u32).map(|k| h(2 * k) | h(2 * k + 1) << 16).collect() };
    let uint = |l: u32| 0xF000 | l;
    let half = |l: u32| 0x3C00 + 5 * l;
    let cases: [(DXGI_FORMAT, u32, Vec<u32>, Vec<u32>); 2] = [
        (DXGI_FORMAT_R16_UINT, 2 | VIEW_INTEGER, pack(&uint), (0..32).map(uint).collect()),
        (DXGI_FORMAT_R16_FLOAT, 2, pack(&half), (0..32).map(half).collect()),
    ];
    for (format, stride, data, want) in cases {
        let views = [View::Buffer { data: &data, stride: Stride::Format(format, 2) }, View::RawUav(32)];
        for (label, words) in run_tables(&shader, &views, vec![stride, 4]) {
            assert_eq!(words[0], want, "{label} {format:?}");
        }
    }
}
/// User data: s2 a table with a raw UAV. Lanes 4k..4k+3 write distinct bytes
/// 0x11, 0x22, 0x33, 0x44 to bytes 0..3 of dword k (`0x44332211`).
const BYTE_STORE_DISTINCT_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3602_0083, // v_and_b32 v1, 3, v0
    0x4A02_0281, // v_add_nc_u32 v1, 1, v1
    0x3404_0284, // v_lshlrev_b32 v2, 4, v1
    0x3804_0501, // v_or_b32 v2, v1, v2
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE060_1000,
    0x8004_0200, // buffer_store_byte v2, v0, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Same as [`BYTE_STORE_DISTINCT_WAVE32`], guarded so only lanes 0..3 execute
/// `buffer_store_byte`; higher dwords must remain untouched.
const BYTE_STORE_GUARDED_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3602_0083, // v_and_b32 v1, 3, v0
    0x4A02_0281, // v_add_nc_u32 v1, 1, v1
    0x3404_0284, // v_lshlrev_b32 v2, 4, v1
    0x3804_0501, // v_or_b32 v2, v1, v2
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7DA8_0084, // v_cmpx_gt_u32 exec_lo, 4, v0
    0xE060_1000,
    0x8004_0200, // buffer_store_byte v2, v0, s[16:19], 0 offen
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Even lanes write 0x1234 to the low
/// half and odd lanes write 0xABCD to the high half of each dword
/// (`0xABCD1234`).
const HALF_STORE_DISTINCT_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3602_0081, // v_and_b32 v1, 1, v0
    0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
    0x7E04_02FF,
    0x0000_ABCD, // v_mov_b32 v2, 0xABCD
    0x7E06_02FF,
    0x0000_1234, // v_mov_b32 v3, 0x1234
    0x0204_0702, // v_cndmask_b32 v2, v2, v3, vcc_lo
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0201, // buffer_store_short v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Lanes 2k and 2k+1 concurrently write
/// disjoint bit patterns `0x55` and `0xAA` to the exact same byte offset k.
/// Under the old AND+OR lowering, lockstep lanes interleave clears and sets
/// to produce `0xFF`; a CAS loop must always leave strictly `0x55` or `0xAA`.
const BYTE_STORE_RACE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3602_0081, // v_and_b32 v1, 1, v0
    0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
    0x7E04_02FF,
    0x0000_00AA, // v_mov_b32 v2, 0xAA
    0x7E06_02FF,
    0x0000_0055, // v_mov_b32 v3, 0x55
    0x0204_0702, // v_cndmask_b32 v2, v2, v3, vcc_lo
    0x2C02_0081, // v_lshrrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE060_1000,
    0x8004_0201, // buffer_store_byte v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Lanes 2k and 2k+1 concurrently write
/// disjoint bit patterns `0x0F0F` and `0xF0F0` to the exact same halfword
/// byte offset `2k`. A CAS loop must always leave strictly `0x0F0F` or
/// `0xF0F0`, never `0xFFFF`.
const HALF_STORE_RACE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3602_0081, // v_and_b32 v1, 1, v0
    0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
    0x7E04_02FF,
    0x0000_F0F0, // v_mov_b32 v2, 0xF0F0
    0x7E06_02FF,
    0x0000_0F0F, // v_mov_b32 v3, 0x0F0F
    0x0204_0702, // v_cndmask_b32 v2, v2, v3, vcc_lo
    0x2C02_0081, // v_lshrrev_b32 v1, 1, v0
    0x3402_0281, // v_lshlrev_b32 v1, 1, v1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0201, // buffer_store_short v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

fn build_subdword_store(words: &[u32], threads: u32) -> HostShader {
    build(
        words,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        threads,
    )
}

/// Dispatches a single-UAV sub-dword store shader over a buffer initialised to
/// `init` on every available SM 6.6 host (hardware GPU and WARP).
fn run_subdword_store(
    shader: &HostShader,
    init: &[u32],
    groups: [u32; 3],
    mut check: impl FnMut(&str, &[u32]),
) {
    let views = [View::Memory(init), View::Pair];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!((u.root, u.slot), (RootType::Table, 2));
            // View::Memory places its raw UAV at heap index 1 (32 bytes in).
            ud[u.index as usize] = 32;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        check(label, &words[0]);
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn subdword_cas_shaders_lower_and_validate() {
    let cases: [(&str, &[u32], u32, &[&str]); 5] = [
        (
            "BYTE_STORE_DISTINCT_WAVE32",
            BYTE_STORE_DISTINCT_WAVE32,
            32,
            &[
                "s_mov_b32 s12, s2",
                "s_mov_b32 s13, 0",
                "s_load_dwordx4 s[16:19], s[12:13], null",
                "v_and_b32 v1, 3, v0",
                "v_add_nc_u32 v1, 1, v1",
                "v_lshlrev_b32 v2, 4, v1",
                "v_or_b32 v2, v1, v2",
                "s_waitcnt lgkmcnt(0)",
                "buffer_store_byte v2, v0, s[16:19], 0 offen",
                "s_endpgm",
            ],
        ),
        (
            "BYTE_STORE_GUARDED_WAVE32",
            BYTE_STORE_GUARDED_WAVE32,
            32,
            &[
                "s_mov_b32 s12, s2",
                "s_mov_b32 s13, 0",
                "s_load_dwordx4 s[16:19], s[12:13], null",
                "v_and_b32 v1, 3, v0",
                "v_add_nc_u32 v1, 1, v1",
                "v_lshlrev_b32 v2, 4, v1",
                "v_or_b32 v2, v1, v2",
                "s_waitcnt lgkmcnt(0)",
                "v_cmpx_gt_u32 exec_lo, 4, v0",
                "buffer_store_byte v2, v0, s[16:19], 0 offen",
                "s_mov_b32 exec_lo, -1",
                "s_endpgm",
            ],
        ),
        (
            "HALF_STORE_DISTINCT_WAVE32",
            HALF_STORE_DISTINCT_WAVE32,
            32,
            &[
                "s_mov_b32 s12, s2",
                "s_mov_b32 s13, 0",
                "s_load_dwordx4 s[16:19], s[12:13], null",
                "v_and_b32 v1, 1, v0",
                "v_cmp_eq_u32 vcc_lo, 0, v1",
                "v_mov_b32 v2, lit(0x0000abcd)",
                "v_mov_b32 v3, lit(0x00001234)",
                "v_cndmask_b32 v2, v2, v3, vcc_lo",
                "v_lshlrev_b32 v1, 1, v0",
                "s_waitcnt lgkmcnt(0)",
                "buffer_store_short v2, v1, s[16:19], 0 offen",
                "s_endpgm",
            ],
        ),
        (
            "BYTE_STORE_RACE_WAVE32",
            BYTE_STORE_RACE_WAVE32,
            64,
            &[
                "s_mov_b32 s12, s2",
                "s_mov_b32 s13, 0",
                "s_load_dwordx4 s[16:19], s[12:13], null",
                "v_and_b32 v1, 1, v0",
                "v_cmp_eq_u32 vcc_lo, 0, v1",
                "v_mov_b32 v2, lit(0x000000aa)",
                "v_mov_b32 v3, lit(0x00000055)",
                "v_cndmask_b32 v2, v2, v3, vcc_lo",
                "v_lshrrev_b32 v1, 1, v0",
                "s_waitcnt lgkmcnt(0)",
                "buffer_store_byte v2, v1, s[16:19], 0 offen",
                "s_endpgm",
            ],
        ),
        (
            "HALF_STORE_RACE_WAVE32",
            HALF_STORE_RACE_WAVE32,
            64,
            &[
                "s_mov_b32 s12, s2",
                "s_mov_b32 s13, 0",
                "s_load_dwordx4 s[16:19], s[12:13], null",
                "v_and_b32 v1, 1, v0",
                "v_cmp_eq_u32 vcc_lo, 0, v1",
                "v_mov_b32 v2, lit(0x0000f0f0)",
                "v_mov_b32 v3, lit(0x00000f0f)",
                "v_cndmask_b32 v2, v2, v3, vcc_lo",
                "v_lshrrev_b32 v1, 1, v0",
                "v_lshlrev_b32 v1, 1, v1",
                "s_waitcnt lgkmcnt(0)",
                "buffer_store_short v2, v1, s[16:19], 0 offen",
                "s_endpgm",
            ],
        ),
    ];
    for (name, words, threads, want_sp3) in cases {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let disasm: Vec<String> = crate::gfx10::decode(&code)
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e}"))
            .iter()
            .map(|inst| inst.text(false))
            .collect();
        assert_eq!(disasm, want_sp3, "{name}: unexpected disassembly");
        let _ = build_subdword_store(words, threads);
    }
}

/// 4 lanes writing distinct bytes (`0x11, 0x22, 0x33, 0x44`) into one dword
/// produce `0x44332211`, 2 lanes writing halves (`0x1234, 0xABCD`) produce
/// `0xABCD1234`, and a lane-guarded byte store leaves untouched dwords intact.
#[test]
fn subdword_buffer_stores_pack_distinct_lanes_in_one_dword() {
    let bytes = build_subdword_store(BYTE_STORE_DISTINCT_WAVE32, 32);
    run_subdword_store(&bytes, &[0xA5A5_A5A5; 8], [1, 1, 1], |label, got| {
        assert_eq!(got, &[0x4433_2211; 8], "{label}: distinct byte lanes");
    });

    let halves = build_subdword_store(HALF_STORE_DISTINCT_WAVE32, 32);
    run_subdword_store(&halves, &[0xFFFF_FFFF; 16], [1, 1, 1], |label, got| {
        assert_eq!(got, &[0xABCD_1234; 16], "{label}: distinct half lanes");
    });

    let guarded = build_subdword_store(BYTE_STORE_GUARDED_WAVE32, 32);
    run_subdword_store(&guarded, &[0xDEAD_BEEF; 2], [1, 1, 1], |label, got| {
        assert_eq!(
            got,
            &[0x4433_2211, 0xDEAD_BEEF],
            "{label}: guarded byte lanes"
        );
    });
}

/// Concurrent lanes writing disjoint bit patterns (`0x55` vs `0xAA` for byte,
/// `0x0F0F` vs `0xF0F0` for short) to the exact same byte or halfword must
/// never tear into `0xFF` or `0xFFFF`.
#[test]
fn subdword_buffer_stores_do_not_tear_concurrent_same_slot_writes() {
    let byte_race = build_subdword_store(BYTE_STORE_RACE_WAVE32, 64);
    run_subdword_store(&byte_race, &[0x3333_3333; 8], [16, 1, 1], |label, got| {
        for (dword, &word) in got.iter().enumerate() {
            for slot in 0..4 {
                let b = (word >> (slot * 8)) & 0xFF;
                assert!(
                    b == 0x55 || b == 0xAA,
                    "{label}: dword {dword} byte {slot} tore to {b:#04x} (word {word:#010x})"
                );
            }
        }
    });

    let half_race = build_subdword_store(HALF_STORE_RACE_WAVE32, 64);
    run_subdword_store(&half_race, &[0x3333_3333; 16], [16, 1, 1], |label, got| {
        for (dword, &word) in got.iter().enumerate() {
            for slot in 0..2 {
                let h = (word >> (slot * 16)) & 0xFFFF;
                assert!(
                    h == 0x0F0F || h == 0xF0F0,
                    "{label}: dword {dword} half {slot} tore to {h:#06x} (word {word:#010x})"
                );
            }
        }
    });
}
