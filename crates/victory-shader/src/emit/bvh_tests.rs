//! The BVH intersection emulation (`lower::bvh`): a CPU reference of the
//! box and triangle tests with their edge cases, and synthetic GFX10
//! programs running `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`
//! over random nodes and rays on the host GPU and WARP, compared with the
//! reference word for word. The nodes come from a root SRV whose address
//! the program shifts into the descriptor, or from a pointer in root
//! constants; both are title memory (`Lower::memory`) in such a program.

use super::bitcode::{Bin, Cast, pred};
use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const INVALID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct Ray {
    extent: f32,
    origin: [f32; 3],
    dir: [f32; 3],
    inv: [f32; 3],
}

impl Ray {
    fn new(origin: [f32; 3], dir: [f32; 3], extent: f32) -> Self {
        Ray {
            extent,
            origin,
            dir,
            inv: dir.map(|d| 1.0 / d),
        }
    }
}

// ---- the reference, in the lowering's order of operations ----

/// The instruction's four result dwords for the node `w` (its dwords, 28 for
/// a box32 node, 16 otherwise) behind a pointer of type `ty`.
fn intersect(w: &[u32], ty: u32, ray: &Ray) -> [u32; 4] {
    match ty & 7 {
        0..=3 => triangle(w, ty & 7, ray),
        4 => boxes(w, true, ray),
        5 => boxes(w, false, ray),
        _ => [INVALID; 4],
    }
}

fn half(bits: u32) -> f32 {
    let bits = bits & 0xFFFF;
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = ((bits >> 10) & 0x1F, bits & 0x3FF);
    sign * match e {
        0 => m as f32 * 2f32.powi(-24),
        0x1F if m == 0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    }
}

fn boxes(w: &[u32], box16: bool, ray: &Ray) -> [u32; 4] {
    let mut child = [INVALID; 4];
    let mut key = [0f32; 4];
    for i in 0..4 {
        let b: [f32; 6] = std::array::from_fn(|j| {
            if box16 {
                half(w[4 + 3 * i + j / 2] >> (16 * (j % 2)))
            } else {
                f32::from_bits(w[4 + 6 * i + j])
            }
        });
        let (mut nan, mut near, mut far) = (false, 0f32, f32::INFINITY);
        for a in 0..3 {
            let (dlo, dhi) = (b[a] - ray.origin[a], b[3 + a] - ray.origin[a]);
            let (tlo, thi) = (dlo * ray.inv[a], dhi * ray.inv[a]);
            let pos = ray.inv[a] >= 0.0;
            let (mut n, mut x) = if pos { (tlo, thi) } else { (thi, tlo) };
            let (dn, dx) = if pos { (dlo, dhi) } else { (dhi, dlo) };
            if dn == 0.0 && n.is_nan() {
                n = f32::NEG_INFINITY;
            }
            if dx == 0.0 && x.is_nan() {
                x = f32::INFINITY;
            }
            nan |= n.is_nan() || x.is_nan();
            if n > near {
                near = n;
            }
            if x < far {
                far = x;
            }
        }
        if ray.extent < far {
            far = ray.extent;
        }
        if !nan && near <= far {
            child[i] = w[i];
        }
        key[i] = near;
    }
    for (a, b) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
        if child[b] != INVALID && key[b] < key[a] || child[a] == INVALID {
            child.swap(a, b);
            key.swap(a, b);
        }
    }
    child
}

fn triangle(w: &[u32], ty: u32, ray: &Ray) -> [u32; 4] {
    let vertex = |k: usize| -> [f32; 3] { std::array::from_fn(|j| f32::from_bits(w[3 * k + j])) };
    let [p0, p1, p2] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]][ty as usize].map(vertex);
    let ad = ray.dir.map(f32::abs);
    let x_ge_y = ad[0] >= ad[1];
    let z0 = x_ge_y && ad[0] >= ad[2];
    let z1 = !x_ge_y && ad[1] >= ad[2];
    let by = |v: [f32; 3], k: [usize; 3]| {
        if z0 {
            v[k[0]]
        } else if z1 {
            v[k[1]]
        } else {
            v[k[2]]
        }
    };
    let neg = by(ray.dir, [0, 1, 2]) < 0.0;
    let perm = |v: [f32; 3]| {
        let (x, y, z) = (by(v, [1, 2, 0]), by(v, [2, 0, 1]), by(v, [0, 1, 2]));
        if neg { [y, x, z] } else { [x, y, z] }
    };
    let d = perm(ray.dir);
    let sz = 1.0 / d[2];
    let (sx, sy) = (d[0] * sz, d[1] * sz);
    let sheared = |p: [f32; 3]| {
        let r = perm(std::array::from_fn(|k| p[k] - ray.origin[k]));
        [r[0] - sx * r[2], r[1] - sy * r[2], sz * r[2]]
    };
    let (a, b, c) = (sheared(p0), sheared(p1), sheared(p2));
    let edge = |p: [f32; 3], q: [f32; 3]| p[0] * q[1] - p[1] * q[0];
    let (u, v, ww) = (edge(c, b), edge(a, c), edge(b, a));
    let inside = u >= 0.0 && v >= 0.0 && ww >= 0.0 || u <= 0.0 && v <= 0.0 && ww <= 0.0;
    let det = u + (v + ww);
    let t = (u * a[2] + v * b[2]) + ww * c[2];
    let ahead = det > 0.0 && t >= 0.0 || det < 0.0 && t <= 0.0;
    if !(inside && ahead) {
        return [f32::INFINITY.to_bits(), 1.0f32.to_bits(), 0, 0];
    }
    let id = w[15] >> (8 * ty);
    let bary = |s: u32| [u, v, ww].get(s as usize & 3).map_or(0, |x| x.to_bits());
    [t.to_bits(), det.to_bits(), bary(id), bary(id >> 2)]
}

// ---- reference tests ----

/// A box32 node of four boxes (min xyz, max xyz) with children 1..=4.
fn box32(boxes: [[f32; 6]; 4]) -> Vec<u32> {
    let mut w = vec![1, 2, 3, 4];
    w.extend(boxes.iter().flatten().map(|x| x.to_bits()));
    w
}

fn unit(at: f32) -> [f32; 6] {
    [at, -1.0, -1.0, at + 1.0, 1.0, 1.0]
}

/// A box nothing hits (min above max).
const EMPTY: [f32; 6] = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

fn along_x() -> Ray {
    Ray::new([0.0; 3], [1.0, 0.0, 0.0], 100.0)
}

#[test]
fn boxes_come_sorted_by_distance_misses_last() {
    let w = box32([unit(30.0), EMPTY, unit(10.0), unit(-5.0)]);
    assert_eq!(boxes(&w, false, &along_x()), [3, 1, INVALID, INVALID]);
    let w = box32([unit(3.0), unit(2.0), unit(1.0), unit(0.0)]);
    assert_eq!(boxes(&w, false, &along_x()), [4, 3, 2, 1]);
}

#[test]
fn equal_distances_keep_the_networks_order() {
    let w = box32([unit(5.0); 4]);
    assert_eq!(boxes(&w, false, &along_x()), [1, 2, 3, 4]);
    // An invalid first child moves, which reorders equal ones.
    let w = box32([EMPTY, unit(5.0), unit(5.0), unit(5.0)]);
    assert_eq!(boxes(&w, false, &along_x()), [3, 2, 4, INVALID]);
    // A box around the origin is at distance 0, as one that starts there.
    let w = box32([unit(0.0), unit(-0.5), EMPTY, EMPTY]);
    assert_eq!(boxes(&w, false, &along_x()), [1, 2, INVALID, INVALID]);
}

#[test]
fn a_ray_parallel_to_a_slab_hits_only_between_its_planes() {
    // Direction 0 along y and z: inverses +-inf.
    for dir in [[1.0, 0.0, 0.0], [1.0, -0.0, -0.0]] {
        let ray = Ray::new([0.0; 3], dir, 100.0);
        let inside = [2.0, -1.0, -1.0, 3.0, 1.0, 1.0];
        let outside = [2.0, 0.5, -1.0, 3.0, 1.0, 1.0];
        // The origin on a plane (0 * inf): conservatively inside.
        let on_min = [2.0, 0.0, -1.0, 3.0, 1.0, 1.0];
        let on_max = [2.0, -1.0, -1.0, 3.0, 1.0, 0.0];
        let w = box32([inside, outside, on_min, on_max]);
        assert_eq!(boxes(&w, false, &ray), [1, 4, 3, INVALID], "{dir:?}");
    }
}

#[test]
fn extent_and_origin_bound_the_interval_inclusively() {
    let ray = Ray::new([0.0; 3], [1.0, 0.0, 0.0], 10.0);
    let w = box32([unit(10.0), unit(10.5), unit(-1.0), unit(-2.0)]);
    // At the extent: a hit; behind the origin, touching it: a hit at 0.
    assert_eq!(boxes(&w, false, &ray), [3, 1, INVALID, INVALID]);
    let w = box32([unit(-3.0), [0.0; 6], EMPTY, EMPTY]);
    // Entirely behind; a point box at the origin.
    assert_eq!(boxes(&w, false, &ray), [2, INVALID, INVALID, INVALID]);
    // A NaN extent bounds nothing.
    let ray = Ray::new([0.0; 3], [1.0, 0.0, 0.0], f32::NAN);
    let w = box32([unit(1e30), EMPTY, EMPTY, EMPTY]);
    assert_eq!(boxes(&w, false, &ray), [1, INVALID, INVALID, INVALID]);
}

#[test]
fn nan_bounds_miss() {
    let mut nan_min = unit(1.0);
    nan_min[0] = f32::NAN;
    let mut nan_max = unit(2.0);
    nan_max[5] = f32::NAN;
    let w = box32([nan_min, nan_max, unit(3.0), EMPTY]);
    assert_eq!(boxes(&w, false, &along_x()), [3, INVALID, INVALID, INVALID]);
    let mut ray = along_x();
    ray.origin[1] = f32::NAN;
    assert_eq!(boxes(&w, false, &ray), [INVALID; 4]);
}

#[test]
fn box16_holds_halves_in_pairs() {
    // minx|miny, minz|maxx, maxy|maxz; 2.0 = 0x4000, 3.0 = 0x4200,
    // -1.0 = 0xBC00, 1.0 = 0x3C00.
    let b = [0xBC00_4000, 0x4200_BC00, 0x3C00_3C00];
    let mut w = vec![7, 8, 9, 10];
    for k in 0..4 {
        w.extend(if k == 1 { b } else { [0x3C00_3C00, 0x3C00, 0] });
    }
    assert_eq!(intersect(&w, 4, &along_x()), [8, INVALID, INVALID, INVALID]);
}

/// A triangle node of `v` as vertices V0..V2 (type 0), triangle id 9.
fn tri0(v: [[f32; 3]; 3]) -> Vec<u32> {
    let mut w: Vec<u32> = v.iter().flatten().map(|x| x.to_bits()).collect();
    w.resize(15, 0);
    w.push(9);
    w
}

fn ratio(r: [u32; 4]) -> (f32, f32, f32, f32) {
    let f = r.map(f32::from_bits);
    (f[0] / f[1], f[2] / f[1], f[3] / f[1], f[1])
}

#[test]
fn a_triangle_returns_t_and_barycentrics_over_one_denominator() {
    let ray = Ray::new([0.25, 0.25, -1.0], [0.0, 0.0, 1.0], 100.0);
    let v = [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    let (t, i, j, det) = ratio(triangle(&tri0(v), 0, &ray));
    assert_eq!((t, i, j), (3.0, 0.25, 0.25));
    // The ray runs along (V1 - V0) x (V2 - V0): a back face.
    assert!(det < 0.0);
    let (_, _, _, det) = ratio(triangle(&tri0([v[0], v[2], v[1]]), 0, &ray));
    assert!(det > 0.0);
    // Behind the origin: a miss; no extent test.
    let mut back = ray;
    back.origin[2] = 3.0;
    assert_eq!(
        triangle(&tri0(v), 0, &back),
        [0x7F80_0000, 0x3F80_0000, 0, 0]
    );
    let mut short = ray;
    short.extent = 1.0;
    assert_eq!(ratio(triangle(&tri0(v), 0, &short)).0, 3.0);
}

#[test]
fn parallel_rays_zero_area_and_nan_miss() {
    let miss = [0x7F80_0000, 0x3F80_0000, 0, 0];
    let v = [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]];
    // In the triangle's plane, through it.
    let ray = Ray::new([-1.0, 0.25, 2.0], [1.0, 0.0, 0.0], 100.0);
    assert_eq!(triangle(&tri0(v), 0, &ray), miss);
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);
    // A degenerate triangle whose segment the ray crosses.
    let line = [[-1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, 2.0]];
    assert_eq!(triangle(&tri0(line), 0, &ray), miss);
    let point = [[0.0, 0.0, 2.0]; 3];
    assert_eq!(triangle(&tri0(point), 0, &ray), miss);
    let mut nan = v;
    nan[1][0] = f32::NAN;
    assert_eq!(triangle(&tri0(nan), 0, &ray), miss);
}

#[test]
fn shared_edges_and_vertices_leave_no_gap() {
    // A fan of triangles around the origin's projection: a ray through the
    // shared vertex or along a shared edge hits on both sides of it.
    let ray = Ray::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 100.0);
    let hub = [0.0, 0.0, 1.0];
    let rim: Vec<[f32; 3]> = (0..6)
        .map(|k| {
            let a = k as f32 * std::f32::consts::FRAC_PI_3;
            [a.cos(), a.sin(), 1.0]
        })
        .collect();
    let mut hits = 0;
    for k in 0..6 {
        let r = triangle(&tri0([hub, rim[k], rim[(k + 1) % 6]]), 0, &ray);
        hits += usize::from(f32::from_bits(r[0]).is_finite());
    }
    assert_eq!(hits, 6);
    // Random points on the shared diagonal of a quad hit at least one half,
    // and the halves agree where both hit.
    let (p, q) = ([0.3f32, -0.7, 1.0], [-0.9f32, 0.6, 1.3]);
    let (s, t) = ([1.1f32, 0.9, 0.7], [-1.2f32, -0.8, 1.1]);
    let mut rng = Rng(0x1234_5678);
    for _ in 0..10_000 {
        let k = rng.unit();
        let at: [f32; 3] = std::array::from_fn(|i| p[i] + k * (q[i] - p[i]));
        let o = [rng.unit() - 0.5, rng.unit() - 0.5, -2.0];
        let d: [f32; 3] = std::array::from_fn(|i| at[i] - o[i]);
        let ray = Ray::new(o, d, 100.0);
        let one = triangle(&tri0([p, q, s]), 0, &ray);
        let two = triangle(&tri0([q, p, t]), 0, &ray);
        let hit = |r: [u32; 4]| f32::from_bits(r[0]).is_finite();
        assert!(hit(one) || hit(two), "{ray:?}");
    }
}

#[test]
fn the_triangle_id_names_vertices_and_barycentrics() {
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);
    // V0..V4: V1, V3, V2 (type 1) and V2, V4, V0 (type 3) cover the origin.
    let v: [[f32; 3]; 5] = [
        [-1.0, 3.0, 1.0],
        [-1.0, -1.0, 1.0],
        [3.0, -1.0, 1.0],
        [1.0, 3.0, 1.0],
        [-4.0, -4.0, 1.0],
    ];
    let mut w: Vec<u32> = v.iter().flatten().map(|x| x.to_bits()).collect();
    // Type 1: I from 2, J from 1; type 3: I from 0, J from 3 (undefined).
    w.push((2 | 1 << 2) << 8 | (3 << 2) << 24);
    let r = triangle(&w, 1, &ray);
    let (b1, b2) = (f32::from_bits(r[2]), f32::from_bits(r[3]));
    let plain = {
        let mut p = w.clone();
        p[15] = 9 << 8;
        triangle(&p, 1, &ray)
    };
    assert_eq!((r[0], r[1]), (plain[0], plain[1]));
    assert_eq!((b1.to_bits(), b2.to_bits()), (plain[3], plain[2]));
    let r = triangle(&w, 3, &ray);
    assert!(f32::from_bits(r[0]).is_finite());
    assert_eq!(r[3], 0);
    // Weights sum to the denominator: the first vertex's from source 0.
    let f = r.map(f32::from_bits);
    let i = triangle(
        &{
            let mut p = w.clone();
            p[15] = (1 | 2 << 2) << 24;
            p
        },
        3,
        &ray,
    )
    .map(f32::from_bits);
    assert!((f[2] + i[2] + i[3] - f[1]).abs() <= 1e-5 * f[1].abs());
    // Types 2 and 3 read V4 from byte 48; types 6 and 7 return nothing.
    assert_eq!(intersect(&w, 6, &ray), [INVALID; 4]);
    assert_eq!(intersect(&w, 7, &ray), [INVALID; 4]);
}

#[test]
fn the_denominators_sign_and_ratios_match_moller_trumbore() {
    let mut rng = Rng(42);
    let mut hits = 0;
    for _ in 0..20_000 {
        let v: [[f32; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.unit() * 4.0 - 2.0));
        let o: [f32; 3] = std::array::from_fn(|_| rng.unit() * 4.0 - 2.0);
        let d: [f32; 3] = std::array::from_fn(|_| rng.unit() * 2.0 - 1.0);
        let ray = Ray::new(o, d, 100.0);
        let r = triangle(&tri0(v), 0, &ray);
        // Moller-Trumbore in f64: det = (d x e2) . e1, the hardware's sign.
        let f = |a: [f32; 3]| a.map(f64::from);
        let (v0, v1, v2, o, d) = (f(v[0]), f(v[1]), f(v[2]), f(o), f(d));
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (e1, e2, e3) = (sub(v1, v0), sub(v2, v0), sub(o, v0));
        let (s1, s2) = (cross(d, e2), cross(e3, e1));
        let det = dot(s1, e1);
        let (t, i, j) = (dot(e2, s2) / det, dot(e3, s1) / det, dot(d, s2) / det);
        let inside = i > 1e-4 && j > 1e-4 && i + j < 1.0 - 1e-4 && t > 1e-4 && det.abs() > 1e-3;
        let outside = i < -1e-4 || j < -1e-4 || i + j > 1.0 + 1e-4 || t < -1e-4;
        let ours = ratio(r);
        if inside {
            hits += 1;
            assert!(ours.0.is_finite(), "{v:?} {ray:?}");
            assert_eq!(ours.3 > 0.0, det > 0.0);
            for (a, b) in [(ours.0, t), (ours.1, i), (ours.2, j)] {
                assert!((f64::from(a) - b).abs() < 1e-3 * (1.0 + b.abs()), "{a} {b}");
            }
        } else if outside {
            assert!(!ours.0.is_finite(), "{v:?} {ray:?}");
        }
    }
    assert!(hits > 500, "{hits}");
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// In [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1 << 24) as f32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// A multiple of 1/64 in [-range, range].
    fn grid(&mut self, range: f32) -> f32 {
        let n = (range * 64.0) as u32;
        (self.below(2 * n + 1) as f32 - n as f32) / 64.0
    }
}

// ---- the lowering on the GPU ----

/// SOP1 `s_mov_b32 sdst, src0`.
fn s_mov(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | u32::from(sdst) << 16 | 0x03 << 8 | u32::from(src0)
}

/// SOP2 `op sdst, src0, src1`.
fn sop2(op: u32, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | op << 23 | u32::from(sdst) << 16 | u32::from(src1) << 8 | u32::from(src0)
}

const LITERAL: u8 = 255;

/// A synthetic wave32 program of 256 threads: lane n reads its ray from the
/// root SRV at s[0:1] (16 dwords at byte 64n: pointer, extent, origin,
/// direction, inverse direction, pointer high), intersects it with a
/// descriptor based at that SRV, and stores the four dwords to the raw UAV
/// at table entry 0 (s2), at byte 16n.
fn program(wide: bool, guarded: bool) -> Vec<u32> {
    let global_load_x4 =
        |vdst: u32, offset: u32| [0x37 << 26 | 0x0E << 18 | 2 << 14 | offset, 1 | vdst << 24];
    let mut w = vec![
        s_mov(3, 0x80),
        // s_load_dwordx4 s[8:11], s[2:3], 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2 >> 1,
        125 << 25,
        // v_lshlrev_b32 v1, 6, v0
        0x1A << 25 | 1 << 17 | 256 << 9 | (128 + 6),
    ];
    for (k, v) in [2, 6, 10].into_iter().enumerate() {
        w.extend(global_load_x4(v, 16 * k as u32));
    }
    w.extend([
        sop2(0x21, 12, 0, 128 + 8),  // s_lshr_b64 s[12:13], s[0:1], 8
        sop2(0x10, 13, 13, LITERAL), // s_or_b32 s13, s13, 0x80000000
        0x8000_0000,
        s_mov(14, 193), // -1
        // Type 8, barycentrics, the page-size hint, no bound.
        s_mov(15, LITERAL),
        8 << 28 | 1 << 24 | 1 << 27 | 0x3FF,
    ]);
    if guarded {
        // Mask exec to a single lane so `bvh` lowers under a lane guard.
        w.push(0x7DA4_0080); // v_cmpx_eq_u32 exec, 0, v0
    }
    // Result v[20:23], descriptor s[12:15].
    let rest = 20 << 8 | (12 >> 2) << 16;
    if wide {
        // NSA: pointer low, high, then the ray from v3.
        let regs = [2u32, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        w.extend([0xF19C_9F07, rest | regs[0]]);
        for k in 0..3 {
            let reg = |j: usize| regs.get(1 + 4 * k + j).map_or(0, |r| r << (8 * j));
            w.push((0..4).map(reg).fold(0, |a, b| a | b));
        }
    } else {
        w.extend([0xF198_9F01, rest | 2]);
    }
    w.extend([
        // v_lshlrev_b32 v1, 4, v0
        0x1A << 25 | 1 << 17 | 256 << 9 | (128 + 4),
        // buffer_store_dwordx4 v[20:23], v1, s[8:11], 0 offen
        0x38 << 26 | 0x1E << 18 | 1 << 12,
        0x80 << 24 | (8 >> 2) << 16 | 20 << 8 | 1,
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(wide: bool, memory: bool) -> HostShader {
    build_with(wide, memory, false)
}

fn build_with(wide: bool, memory: bool, guarded: bool) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: if memory {
                    RootType::Constants
                } else {
                    RootType::Descriptor
                },
                table_offset_dwords: 0,
                sgprs: 2,
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
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
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
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [256, 1, 1],
        }),
        code: program(wide, guarded)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    assert_eq!(
        res.descriptors
            .iter()
            .any(|d| matches!(d.source, resource::Source::Built { .. })),
        !memory,
        "the BVH's base traced to the root SRV, or read from the operand"
    );
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
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
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

const LANES: usize = 256;

/// f16 bits of a multiple of 1/16 in [-64, 64] (exact).
fn to_half(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let e = (bits >> 23 & 0xFF) as i32 - 127 + 15;
    (bits >> 16 & 0x8000) | (e as u32) << 10 | (bits >> 13 & 0x3FF)
}

/// One random case: the node's dwords (28), its pointer type, the ray.
fn case(rng: &mut Rng) -> (Vec<u32>, u32, Ray) {
    case_for_type(rng, None)
}

fn case_for_type(rng: &mut Rng, forced_ty: Option<u32>) -> (Vec<u32>, u32, Ray) {
    // The dominant direction component a power of two (exact shear), the
    // others smaller; now and then 0 (inverse +-inf).
    let kz = rng.below(3) as usize;
    let big =
        [0.5f32, 1.0, 2.0][rng.below(3) as usize] * if rng.below(2) == 0 { 1.0 } else { -1.0 };
    let dir: [f32; 3] = std::array::from_fn(|k| match k {
        _ if k == kz => big,
        _ if rng.below(6) == 0 => [0.0, -0.0][rng.below(2) as usize],
        _ => rng.grid(big.abs()),
    });
    let origin: [f32; 3] = std::array::from_fn(|_| rng.grid(8.0));
    let extent = [f32::INFINITY, 5.0, 40.0, rng.grid(64.0).abs()][rng.below(4) as usize];
    let ray = Ray::new(origin, dir, extent);
    let near = |rng: &mut Rng| -> [f32; 3] {
        let t = rng.grid(24.0);
        std::array::from_fn(|k| ((origin[k] + t * dir[k]) * 16.0).round() / 16.0 + rng.grid(1.0))
    };
    let mut w: Vec<u32> = (0..28).map(|_| rng.next()).collect();
    let ty = match forced_ty {
        Some(ty) => ty,
        None => match rng.below(20) {
            0 => 6 + rng.below(2),
            1..=7 => 5,
            8..=11 => 4,
            _ => rng.below(4),
        },
    };
    match ty {
        4 | 5 => {
            for i in 0..4 {
                let c = near(rng);
                let mut b: [f32; 6] = std::array::from_fn(|j| {
                    let s = rng.grid(2.0);
                    if j < 3 {
                        c[j] - s.abs()
                    } else {
                        c[j - 3] + s.abs()
                    }
                });
                match rng.below(8) {
                    // Inverted: empty.
                    0 => b.swap(0, 3),
                    // A plane through the origin (rays with a zero direction
                    // component lie in it).
                    1 => b[rng.below(6) as usize] = origin[rng.below(3) as usize],
                    _ => {}
                }
                if ty == 4 {
                    let h = b.map(|x| to_half((x * 16.0).round() / 16.0));
                    w[4 + 3 * i] = h[0] | h[1] << 16;
                    w[5 + 3 * i] = h[2] | h[3] << 16;
                    w[6 + 3 * i] = h[4] | h[5] << 16;
                } else {
                    for (j, x) in b.iter().enumerate() {
                        w[4 + 6 * i + j] = x.to_bits();
                    }
                }
            }
        }
        0..=3 => {
            let c = near(rng);
            let mut v: [[f32; 3]; 5] = std::array::from_fn(|_| c.map(|x| x + rng.grid(3.0)));
            if rng.below(2) == 0 {
                // The tested triangle's centroid on the ray: a hit unless behind.
                let [p, q, r] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]][ty as usize];
                v[r] = std::array::from_fn(|k| 3.0 * c[k] - v[p][k] - v[q][k]);
            }
            for (k, x) in v.iter().flatten().enumerate() {
                w[k] = x.to_bits();
            }
            if rng.below(10) == 0 {
                // A zero-area triangle.
                w.copy_within(0..3, 3);
            }
        }
        _ => {}
    }
    (w, ty, ray)
}

fn run_cases(wide: bool, batches: usize, memory: bool) {
    let mut rng = Rng(if wide { 77 } else { 7 });
    let batches: Vec<Vec<(Vec<u32>, u32, Ray)>> = (0..batches)
        .map(|_| (0..LANES).map(|_| case(&mut rng)).collect())
        .collect();
    run_batches(wide, memory, &batches);
}

fn run_batches(wide: bool, memory: bool, batches: &[Vec<(Vec<u32>, u32, Ray)>]) {
    let shader = build(wide, memory);
    // A program with the BVH instruction reads its root SRV as title
    // memory too.
    assert!(shader.memory);
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
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 0,
                (RootType::Constants, 0) if memory => (TITLE >> (32 * u.dword)) as u32,
                (RootType::Descriptor, 0) if !memory => (TITLE >> (32 * u.dword)) as u32,
                _ => panic!("{u:?}"),
            };
        }
        let (mut hits, mut tris) = (0, 0);
        for cases in batches {
            // Rays, then the nodes, 128 bytes each.
            let nodes = (LANES * 64) as u32;
            let mut data = vec![0u32; LANES * 16 + LANES * 32];
            for (n, (w, ty, ray)) in cases.iter().enumerate() {
                let ptr = (nodes + 128 * n as u32) >> 3 | ty;
                let r = &mut data[16 * n..16 * n + 12];
                r[0] = ptr;
                r[1] = ray.extent.to_bits();
                for k in 0..3 {
                    r[2 + k] = ray.origin[k].to_bits();
                    r[5 + k] = ray.dir[k].to_bits();
                    r[8 + k] = ray.inv[k].to_bits();
                }
                let at = LANES * 16 + 32 * n;
                data[at..at + 28].copy_from_slice(w);
            }
            let roots = vec![
                (0, Root::Constants(ud.clone())),
                (1, Root::View { srv: true, data: page_table(1, 3) }),
                (2, Root::Uav(2 * 4096 / 32)),
            ];
            let views = [
                View::RawUav((LANES * 4) as u32),
                View::Memory(&data),
                View::Pair,
                View::NullPair,
                View::Pair,
            ];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots,
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            for (n, (w, ty, ray)) in cases.iter().enumerate() {
                let want = intersect(w, *ty, ray);
                let got = &out[0][4 * n..4 * n + 4];
                assert_eq!(got, want, "{label}: type {ty}, {ray:?}, node {w:08x?}");
                match ty {
                    0..=3 => tris += usize::from(f32::from_bits(want[0]).is_finite()),
                    4 | 5 => hits += want.iter().filter(|&&c| c != INVALID).count(),
                    _ => {}
                }
            }
        }
        eprintln!("{label}: equal ({hits} boxes and {tris} triangles hit)");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn intersections_on_the_gpu_equal_the_reference() {
    run_cases(false, 16, false);
}

#[test]
fn intersections_with_64_bit_pointers_on_the_gpu_equal_the_reference() {
    run_cases(true, 8, false);
}

/// The title address of the rays and nodes in the title-memory runs.
const TITLE: u64 = 0x12_3450_0000;

/// The page table (`emit` "Title memory") for one copy of title memory at
/// [`TITLE`] (under 64 KiB) with its views at heap index `view`, the null
/// pair at `null`.
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    let mut t = vec![0u32; 4096];
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

/// The BVH at a pointer the program has in root constants: nodes (and
/// rays) read through title memory.
#[test]
fn intersections_on_nodes_in_title_memory_equal_the_reference() {
    run_cases(false, 4, true);
    run_cases(true, 4, true);
}

/// Coherent single-type waves (box16, box32, triangle 0, triangles 0..=3,
/// instance, procedural) alongside mixed box16/box32 and fully mixed waves,
/// exercising both uniform fast paths and divergent merges across every
/// node-type branch.
#[test]
fn coherent_and_mixed_node_types_on_the_gpu_equal_the_reference() {
    let mut rng = Rng(0xB74_D15D);
    let make_batch = |rng: &mut Rng| -> Vec<(Vec<u32>, u32, Ray)> {
        (0..LANES)
            .map(|lane| {
                let ty = match lane / 32 {
                    0 => 4,                     // coherent box16
                    1 => 5,                     // coherent box32
                    2 => 0,                     // coherent triangle type 0
                    3 => (lane & 3) as u32,     // coherent triangles (all 4 subtypes)
                    4 => 6,                     // coherent instance
                    5 => 7,                     // coherent procedural
                    6 => 4 + (lane & 1) as u32, // mixed box16 + box32
                    _ => (lane & 7) as u32,     // fully mixed across all 8 types
                };
                case_for_type(rng, Some(ty))
            })
            .collect()
    };
    let batches: Vec<_> = (0..4).map(|_| make_batch(&mut rng)).collect();
    run_batches(false, false, &batches);
    run_batches(true, true, &batches);
}

// ---- zero-GPU DXIL bitcode CFG + validator tests ----

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn read(&mut self, width: u32) -> u64 {
        let mut out = 0u64;
        for bit in 0..width as usize {
            let b = (self.bytes[self.pos >> 3] >> (self.pos & 7)) & 1;
            out |= u64::from(b) << bit;
            self.pos += 1;
        }
        out
    }

    fn vbr(&mut self, width: u32) -> u64 {
        let hi = 1u64 << (width - 1);
        let mut out = 0u64;
        let mut shift = 0u32;
        loop {
            let chunk = self.read(width);
            out |= (chunk & (hi - 1)) << shift;
            if chunk & hi == 0 {
                return out;
            }
            shift += width - 1;
        }
    }

    fn align32(&mut self) {
        self.pos = (self.pos + 31) & !31;
    }

    fn skip_words(&mut self, words: u64) {
        self.pos += words as usize * 32;
    }
}

fn decode_signed(v: u64) -> i64 {
    if v & 1 != 0 {
        -((v >> 1) as i64)
    } else {
        (v >> 1) as i64
    }
}

#[derive(Debug, Clone)]
enum BcType {
    Void,
    Float,
    Struct(Vec<usize>),
    Array(usize),
    Func { ret: usize },
    Other,
}

#[derive(Debug, Clone, Copy)]
enum BcValTy {
    Ty(usize),
    Other,
}

#[derive(Debug)]
enum BcInst {
    Bin {
        op: u64,
        is_float: bool,
        lhs: usize,
        rhs: usize,
    },
    Cast {
        op: u64,
        dst_is_float: bool,
    },
    Cmp {
        pred: u8,
        lhs: usize,
        rhs: usize,
    },
    Phi {
        is_float: bool,
        incoming: Vec<(usize, usize)>,
    },
    Call {
        callee: String,
    },
    Other,
}

#[derive(Debug, Clone, Copy)]
enum BcTerm {
    Ret,
    Br(usize),
    CondBr { cond: usize, t: usize, f: usize },
}

#[derive(Debug, Default)]
struct BcBlock {
    insts: Vec<BcInst>,
    term: Option<BcTerm>,
}

struct DecodedModule {
    blocks: Vec<BcBlock>,
    int_consts: std::collections::HashMap<usize, i64>,
    inst_by_id: std::collections::HashMap<usize, (usize, usize)>,
}

impl DecodedModule {
    fn inst(&self, id: usize) -> &BcInst {
        let &(b, i) = self.inst_by_id.get(&id).expect("instruction value id");
        &self.blocks[b].insts[i]
    }

    fn succs(&self, b: usize) -> Vec<usize> {
        match self.blocks[b].term.expect("terminated block") {
            BcTerm::Ret => Vec::new(),
            BcTerm::Br(t) => vec![t],
            BcTerm::CondBr { t, f, .. } => vec![t, f],
        }
    }

    fn preds(&self, target: usize) -> Vec<usize> {
        (0..self.blocks.len())
            .filter(|&b| self.succs(b).contains(&target))
            .collect()
    }

    fn can_reach(&self, start: usize, goal: usize) -> bool {
        let mut seen = vec![false; self.blocks.len()];
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(b) = stack.pop() {
            if b == goal {
                return true;
            }
            for s in self.succs(b) {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
        }
        false
    }
}

/// Decodes our emitter's unabbreviated LLVM 3.7 bitcode from a DXIL container.
fn decode_dxil_module(container_bytes: &[u8]) -> DecodedModule {
    let container = crate::dxbc::Container::parse(container_bytes).expect("valid DXBC");
    let part = container
        .part(crate::dxbc::FourCC::DXIL)
        .expect("DXIL part");
    assert_eq!(&part.data[8..12], b"DXIL");
    let bc_off = 8 + u32::from_le_bytes(part.data[16..20].try_into().unwrap()) as usize;
    let bc_len = u32::from_le_bytes(part.data[20..24].try_into().unwrap()) as usize;
    let bitcode = &part.data[bc_off..bc_off + bc_len];
    assert_eq!(&bitcode[..4], b"BC\xC0\xDE");

    let mut r = BitReader {
        bytes: bitcode,
        pos: 32,
    };
    assert_eq!(r.read(2), 1, "expected MODULE_BLOCK");
    assert_eq!(r.vbr(8), 8, "expected MODULE_BLOCK id");
    let mod_width = r.vbr(4) as u32;
    r.align32();
    let _mod_words = r.read(32);

    let mut types = Vec::new();
    let mut val_ty = Vec::new();
    let mut int_consts = std::collections::HashMap::new();
    let mut names = std::collections::HashMap::new();
    let mut blocks = Vec::new();
    let mut inst_by_id = std::collections::HashMap::new();

    let field_ty = |types: &[BcType], agg: BcValTy, idx: usize| -> BcValTy {
        match agg {
            BcValTy::Ty(t) => match &types[t] {
                BcType::Struct(fields) => BcValTy::Ty(fields[idx]),
                BcType::Array(elem) => BcValTy::Ty(*elem),
                _ => BcValTy::Other,
            },
            BcValTy::Other => BcValTy::Other,
        }
    };

    loop {
        match r.read(mod_width) {
            0 => {
                r.align32();
                break;
            }
            1 => {
                let id = r.vbr(8);
                let width = r.vbr(4) as u32;
                r.align32();
                let words = r.read(32);
                match id {
                    17 => loop {
                        match r.read(width) {
                            0 => {
                                r.align32();
                                break;
                            }
                            3 => {
                                let code = r.vbr(6);
                                let n = r.vbr(6);
                                let ops: Vec<u64> = (0..n).map(|_| r.vbr(6)).collect();
                                match code {
                                    1 | 19 => {}
                                    2 => types.push(BcType::Void),
                                    3 => types.push(BcType::Float),
                                    11 => types.push(BcType::Array(ops[1] as usize)),
                                    18 | 20 => types.push(BcType::Struct(
                                        ops[1..].iter().map(|&x| x as usize).collect(),
                                    )),
                                    21 => types.push(BcType::Func {
                                        ret: ops[1] as usize,
                                    }),
                                    _ => types.push(BcType::Other),
                                }
                            }
                            other => panic!("unexpected TYPE_BLOCK entry {other}"),
                        }
                    },
                    11 => {
                        let mut cur_ty = 0usize;
                        loop {
                            match r.read(width) {
                                0 => {
                                    r.align32();
                                    break;
                                }
                                3 => {
                                    let code = r.vbr(6);
                                    let n = r.vbr(6);
                                    let ops: Vec<u64> = (0..n).map(|_| r.vbr(6)).collect();
                                    match code {
                                        1 => cur_ty = ops[0] as usize,
                                        4 => {
                                            int_consts.insert(val_ty.len(), decode_signed(ops[0]));
                                            val_ty.push(BcValTy::Ty(cur_ty));
                                        }
                                        3 | 6 | 7 => val_ty.push(BcValTy::Ty(cur_ty)),
                                        _ => panic!("unexpected CONSTANTS_BLOCK code {code}"),
                                    }
                                }
                                other => panic!("unexpected CONSTANTS_BLOCK entry {other}"),
                            }
                        }
                    }
                    14 => loop {
                        match r.read(width) {
                            0 => {
                                r.align32();
                                break;
                            }
                            3 => {
                                let code = r.vbr(6);
                                let n = r.vbr(6);
                                let ops: Vec<u64> = (0..n).map(|_| r.vbr(6)).collect();
                                if code == 1 {
                                    let s: String =
                                        ops[1..].iter().map(|&c| c as u8 as char).collect();
                                    names.insert(ops[0] as usize, s);
                                }
                            }
                            other => panic!("unexpected VALUE_SYMTAB_BLOCK entry {other}"),
                        }
                    },
                    12 => {
                        let mut cur_blk = 0usize;
                        loop {
                            match r.read(width) {
                                0 => {
                                    r.align32();
                                    break;
                                }
                                3 => {
                                    let code = r.vbr(6);
                                    let n = r.vbr(6);
                                    let ops: Vec<u64> = (0..n).map(|_| r.vbr(6)).collect();
                                    let next_id = val_ty.len();
                                    let rel = |x: u64| next_id - x as usize;
                                    match code {
                                        1 => {
                                            blocks = (0..ops[0] as usize)
                                                .map(|_| BcBlock::default())
                                                .collect();
                                        }
                                        2 => {
                                            let (lhs, rhs, op) = (rel(ops[0]), rel(ops[1]), ops[2]);
                                            let ty = val_ty[lhs];
                                            let is_float = matches!(
                                                ty,
                                                BcValTy::Ty(t) if matches!(types[t], BcType::Float)
                                            );
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Bin {
                                                op,
                                                is_float,
                                                lhs,
                                                rhs,
                                            });
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(ty);
                                        }
                                        3 => {
                                            let (dst_ty, op) = (ops[1] as usize, ops[2]);
                                            let dst_is_float =
                                                matches!(types[dst_ty], BcType::Float);
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk]
                                                .insts
                                                .push(BcInst::Cast { op, dst_is_float });
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(BcValTy::Ty(dst_ty));
                                        }
                                        10 => {
                                            blocks[cur_blk].term = Some(BcTerm::Ret);
                                            cur_blk += 1;
                                        }
                                        11 => {
                                            blocks[cur_blk].term = Some(if ops.len() == 1 {
                                                BcTerm::Br(ops[0] as usize)
                                            } else {
                                                BcTerm::CondBr {
                                                    cond: rel(ops[2]),
                                                    t: ops[0] as usize,
                                                    f: ops[1] as usize,
                                                }
                                            });
                                            cur_blk += 1;
                                        }
                                        16 => {
                                            let ty = ops[0] as usize;
                                            let is_float = matches!(types[ty], BcType::Float);
                                            let incoming = ops[1..]
                                                .as_chunks::<2>()
                                                .0
                                                .iter()
                                                .map(|c| {
                                                    let v = (next_id as i64 - decode_signed(c[0]))
                                                        as usize;
                                                    (v, c[1] as usize)
                                                })
                                                .collect();
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk]
                                                .insts
                                                .push(BcInst::Phi { is_float, incoming });
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(BcValTy::Ty(ty));
                                        }
                                        20 => {
                                            let ty = ops[1] as usize;
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(BcValTy::Ty(ty));
                                        }
                                        26 => {
                                            let ty = field_ty(
                                                &types,
                                                val_ty[rel(ops[0])],
                                                ops[1] as usize,
                                            );
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(ty);
                                        }
                                        28 => {
                                            let (lhs, rhs, pred) =
                                                (rel(ops[0]), rel(ops[1]), ops[2] as u8);
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Cmp {
                                                pred,
                                                lhs,
                                                rhs,
                                            });
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(BcValTy::Other);
                                        }
                                        29 => {
                                            let ty = val_ty[rel(ops[0])];
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(ty);
                                        }
                                        34 => {
                                            let fty = ops[2] as usize;
                                            let fn_id = rel(ops[3]);
                                            let callee =
                                                names.get(&fn_id).cloned().unwrap_or_default();
                                            let BcType::Func { ret } = types[fty] else {
                                                panic!("call target not a function type");
                                            };
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Call { callee });
                                            if !matches!(types[ret], BcType::Void) {
                                                inst_by_id.insert(next_id, (cur_blk, idx));
                                                val_ty.push(BcValTy::Ty(ret));
                                            }
                                        }
                                        38 => {
                                            let ty = val_ty[rel(ops[1])];
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(ty);
                                        }
                                        43 | 46 => {
                                            let idx = blocks[cur_blk].insts.len();
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                            inst_by_id.insert(next_id, (cur_blk, idx));
                                            val_ty.push(BcValTy::Other);
                                        }
                                        44 => {
                                            blocks[cur_blk].insts.push(BcInst::Other);
                                        }
                                        other => panic!("unexpected FUNCTION_BLOCK code {other}"),
                                    }
                                }
                                other => panic!("unexpected FUNCTION_BLOCK entry {other}"),
                            }
                        }
                    }
                    _ => r.skip_words(words),
                }
            }
            3 => {
                let code = r.vbr(6);
                let n = r.vbr(6);
                for _ in 0..n {
                    let _ = r.vbr(6);
                }
                if matches!(code, 7 | 8) {
                    val_ty.push(BcValTy::Other);
                }
            }
            other => panic!("unexpected MODULE_BLOCK entry {other}"),
        }
    }

    DecodedModule {
        blocks,
        int_consts,
        inst_by_id,
    }
}

/// Zero-GPU bitcode + `dxil.dll` validator proof that `bvh_lanes` dispatches
/// box, triangle, and non-intersected types into mutually exclusive blocks,
/// keeps all node loads before the dispatch branch, guards `box16`
/// `legacyF16ToF32` unpacks behind `is16`, and hoists ray sign checks.
#[test]
fn bvh_dispatch_branches_by_node_type_and_guards_box16() {
    for (wide, memory, guarded) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (false, false, true),
        (true, true, true),
    ] {
        let shader = build_with(wide, memory, guarded);
        if let Some(path) = std::env::var_os("VICTORY_DXIL") {
            let mut v = validator::Validator::load(Some(std::path::Path::new(&path)))
                .expect("VICTORY_DXIL failed to load");
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let m = decode_dxil_module(&shader.dxil);

        // 1. All 24 `legacyF16ToF32` unpacks live together in one `box16` block.
        let box16_blks: Vec<usize> = (0..m.blocks.len())
            .filter(|&b| {
                m.blocks[b].insts.iter().any(|inst| {
                    matches!(inst, BcInst::Call { callee } if callee == "dx.op.legacyF16ToF32")
                })
            })
            .collect();
        assert_eq!(
            box16_blks.len(),
            1,
            "expected a single box16 unpack block (wide={wide}, memory={memory}, guarded={guarded})"
        );
        let box16_blk = box16_blks[0];
        let f16_calls = m.blocks[box16_blk]
            .insts
            .iter()
            .filter(
                |inst| matches!(inst, BcInst::Call { callee } if callee == "dx.op.legacyF16ToF32"),
            )
            .count();
        assert_eq!(f16_calls, 24, "box16 block should unpack all 24 bounds");

        // 2. `box16_blk` is guarded directly by `is16` (`icmp eq ty, 4`) against `box32_blk`.
        let box16_preds = m.preds(box16_blk);
        assert_eq!(box16_preds.len(), 1);
        let box_head = box16_preds[0];
        let Some(BcTerm::CondBr {
            cond: is16_id,
            t,
            f: box32_blk,
        }) = m.blocks[box_head].term
        else {
            panic!("box_head must end in CondBr(is16, box16, box32)");
        };
        assert_eq!(t, box16_blk);
        assert_ne!(box32_blk, box16_blk);
        assert!(!m.can_reach(box16_blk, box32_blk) && !m.can_reach(box32_blk, box16_blk));
        let &BcInst::Cmp {
            pred: pred::EQ,
            lhs: ty_id,
            rhs: c4_id,
        } = m.inst(is16_id)
        else {
            panic!("is16 must be `icmp eq ty, 4`");
        };
        assert_eq!(m.int_consts.get(&c4_id), Some(&4));

        // `box32_blk` bitcasts 24 f32 bounds and runs zero `legacyF16ToF32` calls.
        assert!(
            !m.blocks[box32_blk].insts.iter().any(|inst| {
                matches!(inst, BcInst::Call { callee } if callee == "dx.op.legacyF16ToF32")
            }),
            "box32 arm must not call legacyF16ToF32"
        );
        let f32_bitcasts = m.blocks[box32_blk]
            .insts
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    BcInst::Cast { op, dst_is_float: true } if *op == Cast::Bitcast as u64
                )
            })
            .count();
        assert_eq!(f32_bitcasts, 24, "box32 block should bitcast 24 bounds");

        // 3. Both box16 and box32 join at `slab_blk` with 24 f32 phis and 3 hoisted OGE checks.
        let (Some(BcTerm::Br(slab_blk)), Some(BcTerm::Br(slab_from_32))) =
            (m.blocks[box16_blk].term, m.blocks[box32_blk].term)
        else {
            panic!("box16 and box32 arms must unconditionally branch to slab join");
        };
        assert_eq!(slab_blk, slab_from_32);
        let slab_f32_phis: Vec<_> = m.blocks[slab_blk]
            .insts
            .iter()
            .filter_map(|inst| match inst {
                BcInst::Phi {
                    is_float: true,
                    incoming,
                } => Some(incoming),
                _ => None,
            })
            .collect();
        assert_eq!(
            slab_f32_phis.len(),
            24,
            "slab block must merge 24 f32 bounds"
        );
        for incoming in slab_f32_phis {
            let mut from: Vec<usize> = incoming.iter().map(|&(_, b)| b).collect();
            from.sort_unstable();
            let mut want = vec![box16_blk, box32_blk];
            want.sort_unstable();
            assert_eq!(from, want);
        }
        let slab_oge = m.blocks[slab_blk]
            .insts
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    BcInst::Cmp {
                        pred: pred::OGE,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            slab_oge, 3,
            "ray sign check (ray.inv[a] >= 0) must be hoisted to 3 OGE compares"
        );

        // 4. `bvh_triangle` (`fdiv` + edge cross products) lives in its own block,
        //    mutually exclusive with every `bvh_box` block.
        let tri_blks: Vec<usize> = (0..m.blocks.len())
            .filter(|&b| {
                m.blocks[b].insts.iter().any(|inst| {
                    matches!(
                        inst,
                        BcInst::Bin { op, is_float: true, .. } if *op == Bin::SDiv as u64
                    )
                })
            })
            .collect();
        assert_eq!(tri_blks.len(), 1, "expected a single triangle block");
        let tri_blk = tri_blks[0];
        for box_blk in [box_head, box16_blk, box32_blk, slab_blk] {
            assert_ne!(box_blk, tri_blk);
            assert!(
                !m.can_reach(box_blk, tri_blk) && !m.can_reach(tri_blk, box_blk),
                "box block {box_blk} and triangle block {tri_blk} must be mutually exclusive"
            );
        }
        let tri_fmuls = m.blocks[tri_blk]
            .insts
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    BcInst::Bin { op, is_float: true, .. } if *op == Bin::Mul as u64
                )
            })
            .count();
        assert!(
            tri_fmuls >= 15,
            "triangle block should contain shear and cross products"
        );

        // 5. `dispatch_blk` branches on `is_box` -> `box_head` / `other_blk`;
        //    `other_blk` branches on `is_tri` -> `tri_blk` / `join_blk`;
        //    `join_blk` merges `(slab_blk, tri_blk, other_blk)` with 4 i32 phis.
        let box_head_preds = m.preds(box_head);
        assert_eq!(box_head_preds.len(), 1);
        let dispatch_blk = box_head_preds[0];
        let Some(BcTerm::CondBr {
            cond: is_box_id,
            t: box_t,
            f: other_blk,
        }) = m.blocks[dispatch_blk].term
        else {
            panic!("dispatch_blk must end in CondBr(is_box, box_head, other_blk)");
        };
        assert_eq!(box_t, box_head);
        let &BcInst::Cmp {
            pred: pred::EQ,
            lhs: t6_id,
            rhs: c4_box,
        } = m.inst(is_box_id)
        else {
            panic!("is_box must be `icmp eq (ty & 6), 4`");
        };
        assert_eq!(m.int_consts.get(&c4_box), Some(&4));
        let &BcInst::Bin {
            op: t6_op,
            is_float: false,
            lhs: t6_ty,
            rhs: c6_id,
        } = m.inst(t6_id)
        else {
            panic!("t6 must be `and ty, 6`");
        };
        assert_eq!(t6_op, Bin::And as u64);
        assert_eq!(t6_ty, ty_id);
        assert_eq!(m.int_consts.get(&c6_id), Some(&6));

        let Some(BcTerm::CondBr {
            cond: is_tri_id,
            t: tri_t,
            f: join_blk,
        }) = m.blocks[other_blk].term
        else {
            panic!("other_blk must end in CondBr(is_tri, tri_blk, join_blk)");
        };
        assert_eq!(tri_t, tri_blk);
        let &BcInst::Cmp {
            pred: pred::IULT,
            lhs: tri_ty,
            rhs: c4_tri,
        } = m.inst(is_tri_id)
        else {
            panic!("is_tri must be `icmp ult ty, 4`");
        };
        assert_eq!(tri_ty, ty_id);
        assert_eq!(m.int_consts.get(&c4_tri), Some(&4));

        assert!(matches!(m.blocks[slab_blk].term, Some(BcTerm::Br(j)) if j == join_blk));
        assert!(matches!(m.blocks[tri_blk].term, Some(BcTerm::Br(j)) if j == join_blk));

        let join_i32_phis: Vec<_> = m.blocks[join_blk]
            .insts
            .iter()
            .filter_map(|inst| match inst {
                BcInst::Phi {
                    is_float: false,
                    incoming,
                } => Some(incoming),
                _ => None,
            })
            .collect();
        assert_eq!(
            join_i32_phis.len(),
            4,
            "join block must merge 4 i32 result dwords"
        );
        let mut want_preds = vec![slab_blk, tri_blk, other_blk];
        want_preds.sort_unstable();
        for incoming in join_i32_phis {
            let mut from: Vec<usize> = incoming.iter().map(|&(_, b)| b).collect();
            from.sort_unstable();
            assert_eq!(from, want_preds);
        }

        // 6. WARP safety: no memory loads or title-memory atomics live inside
        //    any of the new ALU dispatch blocks.
        for alu_blk in [box_head, box16_blk, box32_blk, slab_blk, other_blk, tri_blk] {
            for inst in &m.blocks[alu_blk].insts {
                if let BcInst::Call { callee } = inst {
                    assert!(
                        !callee.contains("Load") && !callee.contains("atomic"),
                        "memory op {callee} must stay outside ALU dispatch block {alu_blk}"
                    );
                }
            }
        }
    }
}
