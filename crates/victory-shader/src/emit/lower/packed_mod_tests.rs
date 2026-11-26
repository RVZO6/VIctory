//! Regression tests verifying `Lower::packed16` fails closed on unmodeled
//! non-VOP3P modifiers (`sdwa`, `abs`, and `v_pk_fmac_f16` `neg`/`opsel`)
//! instead of silently ignoring them or negating only the low f16 half, while
//! keeping plain `v_pk_fmac_f16` and VOP3P `v_pk_fma_f16` modifiers intact.

use super::{Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// Wraps a packed-f16 instruction sequence writing `v2` into a minimal wave32
/// compute shader that stores `v2` to a raw UAV so dead-code elimination does
/// not drop the instruction.
fn lower_packed(inst_words: &[u32]) -> Result<super::Lowered, ShaderError> {
    let mut words = vec![
        0xBE8C_0302u32, // s_mov_b32 s12, s2
        0xBE8D_0380,    // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000,    // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0x7E02_0280,    // v_mov_b32 v1, 0
        0x7E04_0280,    // v_mov_b32 v2, 0
        0xBF8C_C07F,    // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(inst_words);
    words.extend([
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 6,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 2,
            slot: 2,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower(&Parts {
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
}

#[test]
fn packed16_rejects_unmodeled_fmac_and_sdwa_modifiers_while_keeping_valid_ops() {
    // Controls that MUST lower cleanly:
    // 1. Plain VOP2 v_pk_fmac_f16 v2, v0, v1
    let plain_vop2 = (0x3Cu32 << 25) | (2 << 17) | (1 << 9) | 256;
    assert!(
        lower_packed(&[plain_vop2]).is_ok(),
        "plain VOP2 v_pk_fmac_f16 must lower"
    );

    // 2. Plain VOP3 v_pk_fmac_f16 v2, v0, v1 (opcode 0x13C, no abs/neg/opsel)
    let vop3_w0 = 0xD400_0000u32 | (0x13C << 16) | 2;
    let vop3_w1 = (257u32 << 9) | 256;
    assert!(
        lower_packed(&[vop3_w0, vop3_w1]).is_ok(),
        "plain VOP3 v_pk_fmac_f16 must lower"
    );

    // 3. VOP3P v_pk_fma_f16 v2, -v0, v1, v2 with neg_lo and neg_hi (valid VOP3P)
    let vop3p_w0 = 0xCC00_0000u32 | (0x0E << 16) | (1 << 8) | 2; // neg_hi bit 0
    let vop3p_w1 = (1u32 << 29) | (258 << 18) | (257 << 9) | 256; // neg_lo bit 0
    let ok = lower_packed(&[vop3p_w0, vop3p_w1]).expect("VOP3P v_pk_fma_f16 with neg/neg_hi must lower");
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut ok.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _ = ok;

    // Unmodeled modifiers that MUST be refused rather than silently miscompiled:
    // 1. VOP3 v_pk_fmac_f16 with neg=1 (would only negate low f16 half, leaving high untouched)
    assert!(
        lower_packed(&[vop3_w0, vop3_w1 | (1 << 29)]).is_err(),
        "VOP3 v_pk_fmac_f16 with neg must be rejected"
    );

    // 2. VOP3 v_pk_fmac_f16 with abs=1 (silently ignored before fix)
    assert!(
        lower_packed(&[vop3_w0 | (1 << 8), vop3_w1]).is_err(),
        "VOP3 v_pk_fmac_f16 with abs must be rejected"
    );

    // 3. VOP3 v_pk_fmac_f16 with opsel=1 (silently ignored before fix)
    assert!(
        lower_packed(&[vop3_w0 | (1 << 11), vop3_w1]).is_err(),
        "VOP3 v_pk_fmac_f16 with opsel must be rejected"
    );

    // 4. VOP2 v_pk_fmac_f16 with SDWA (0xF9, silently ignored before fix)
    let sdwa_w0 = (0x3Cu32 << 25) | (2 << 17) | (1 << 9) | 0xF9;
    let sdwa_w1 = 0x0606_0600u32; // dst_sel=DWORD, src0_sel=DWORD, src1_sel=DWORD, v0
    assert!(
        lower_packed(&[sdwa_w0, sdwa_w1]).is_err(),
        "VOP2 v_pk_fmac_f16 with SDWA must be rejected"
    );
}
