//! Regression tests ensuring typed `buffer_store_format_xyzw` refuses
//! non-zero `soffset` (scalar register or inline constant), matching
//! `buffer_load_format_*`, rather than silently discarding the offset.

use super::{Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

fn lower_typed_store(soffset: u8) -> Result<super::Lowered, ShaderError> {
    // s0: root constant (also used as non-zero soffset when soffset == 0)
    // s2: table holding a typed buffer UAV at entry 0
    let words = [
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E08_0280, // v_mov_b32 v4, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE8E_0390, // s_mov_b32 s14, 16
        // buffer_store_format_xyzw v[1:4], v0, s[16:19], soffset idxen
        // opcode 0x007 << 18 = 0x001C_0000, idxen = 1 << 13 = 0x2000
        0xE01C_2000,
        (u32::from(soffset) << 24) | (4 << 16) | (1 << 8),
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 6,
        layout: vec![
            RootParam {
                root: RootType::Constants,
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 16,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
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
fn typed_buffer_store_accepts_zero_in_a_scalar_register() {
    // The fixture loads s13 with zero before the store.
    let via_register = lower_typed_store(0x0d).expect("s13 holds zero");
    let inline = lower_typed_store(0x80).expect("inline zero");
    assert_eq!(via_register.dxil, inline.dxil);
    assert!(
        lower_typed_store(0x0e).is_err(),
        "s14 holds a nonzero offset"
    );
}

#[test]
fn typed_buffer_store_rejects_nonzero_soffset_and_accepts_zero() {
    // Control: soffset = inline 0 (0x80) lowers cleanly and validates.
    let ok = lower_typed_store(0x80).expect("zero soffset typed store must lower");
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut ok.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _ = ok;

    // Non-zero inline constant soffset = 16 (0x80 + 16 = 0x90) must be refused.
    assert!(
        lower_typed_store(0x90).is_err(),
        "typed buffer store with inline soffset=16 must not silently drop the offset"
    );

    // Dynamic SGPR soffset = s0 (0x00) must be refused.
    assert!(
        lower_typed_store(0x00).is_err(),
        "typed buffer store with SGPR soffset=s0 must not silently drop the offset"
    );
}
