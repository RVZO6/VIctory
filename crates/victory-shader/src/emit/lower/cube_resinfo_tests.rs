//! Regression test verifying `image_get_resinfo` on `SQ_RSRC_IMG_CUBE`
//! (`dim == 3`, `kind == 5`) reports 6 faces in channel 2 (depth) rather than
//! 1, matching AMD GCN3 §12.16, RDNA3 §10.1, LLPC, Mesa RADV, and gfx1031
//! hardware measurements (companion to #1256 for cube arrays).

use super::super::host::{Dispatch, Host, Root, View};
use super::super::{HostShader, StaticSampler, validator};
use super::{Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Wave32 compute shader querying `image_get_resinfo dmask:0xf dim:SQ_RSRC_IMG_CUBE`
/// on the cube SRV at table user data `s0` for mip levels 0 and 1, writing the
/// 4 result dwords `{width, height, depth/faces, mip_levels}` for mip 0 to UAV
/// dwords 0..4 and mip 1 to UAV dwords 4..8 (table user data `s2`).
const CUBE_RESINFO_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    // Mip 0:
    0x7E02_0280, // v_mov_b32 v1, 0
    // MIMG: opcode 0x00e (image_get_resinfo) << 18 = 0x0038_0000,
    // dmask 0xf << 8 = 0x0F00, dim 3 (CUBE) << 3 = 0x0018.
    // w1: srsrc=s[8:15] (2 << 16), vdata=v[4:7] (4 << 8), vaddr=v1 (1).
    0xF038_0F18,
    0x0002_0401, // image_get_resinfo v[4:7], v1, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_CUBE
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E04_0280, // v_mov_b32 v2, 0
    0xE078_1000,
    0x8004_0402, // buffer_store_dwordx4 v[4:7], v2, s[16:19], 0 offen
    // Mip 1:
    0x7E02_0281, // v_mov_b32 v1, 1
    0xF038_0F18,
    0x0002_0401, // image_get_resinfo v[4:7], v1, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_CUBE
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E04_0290, // v_mov_b32 v2, 16
    0xE078_1000,
    0x8004_0402, // buffer_store_dwordx4 v[4:7], v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

fn build_cube_resinfo() -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [1, 1, 1],
        }),
        code: CUBE_RESINFO_WAVE32
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower(&Parts {
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
    }
    shader
}

#[test]
fn cube_resinfo_dxil_validates() {
    let shader = build_cube_resinfo();
    assert!(!shader.dxil.is_empty());
}

#[test]
fn cube_image_get_resinfo_reports_six_faces() {
    let shader = build_cube_resinfo();
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    // 6 faces * (8*8 + 4*4 + 2*2) = 504 texels
    let texels = vec![[0.0f32; 4]; 6 * (64 + 16 + 4)];
    let views = [View::MippedCube(8, 3, &texels), View::RawUav(8)];
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Table, 0) => 0,
            (RootType::Table, 2) => 32,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud.clone()))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0][0..4], [8, 8, 6, 3], "{label} mip 0");
        assert_eq!(out[0][4..8], [4, 4, 6, 3], "{label} mip 1");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
