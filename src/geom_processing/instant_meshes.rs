// Instant Meshes
/*

Based off of Wenzel Jakob's Instant Meshes, Python Version
*/

use crate::aabb::AABB;
use crate::{F, FaceKind, add, cross, divk, dot, kmul, length, normalize, sub};
use core::ops::Neg;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Args {
    pub orientation_smoothing_iters: usize,
    pub pos_smoothing_iters: usize,
    pub rounding_mode: RoundMode,
    pub scale: F,
}
impl Default for Args {
    fn default() -> Self {
        Args {
            orientation_smoothing_iters: 100,
            pos_smoothing_iters: 100,
            rounding_mode: RoundMode::Round,
            scale: 0.01,
        }
    }
}

pub fn instant_mesh(
    v: &[[F; 3]],
    f: &[FaceKind],
    vert_normals: &[[F; 3]],
    field: &mut [[F; 3]],
    mut rand: impl FnMut() -> F,

    args: &Args,
) -> Vec<[F; 3]> {
    if v.is_empty() || f.is_empty() {
        return vec![];
    }
    assert_eq!(field.len(), v.len());
    let nv = v.len();

    let aabb = AABB::from_slice(v);
    let mut orientation_field = vec![[0.; 3]; nv];
    let mut pos_field = vec![[0.; 3]; nv];

    for (i, &vec) in field.iter().enumerate() {
        orientation_field[i] = normalize(vec);
    }

    for p in &mut pos_field {
        let t = rand().fract();
        *p = add(kmul(1. - t, aabb.min), kmul(t, aabb.max));
    }

    let vv_adj = crate::adjacency::vertex_vertex_adj(nv, f);

    // smooth orientation field
    println!("Smoothing orientation field");
    for _it in 0..args.orientation_smoothing_iters {
        for vi in 0..nv {
            let mut o = orientation_field[vi];
            let n = vert_normals[vi];
            let mut w = 0;
            for &adj_vi in vv_adj.adj(vi) {
                let adj_vi = adj_vi as usize;
                let [o_base, compat] = compat_orientation_extrinsic(
                    o,
                    n,
                    orientation_field[adj_vi],
                    vert_normals[adj_vi],
                );
                o = add(kmul(w as F, o_base), compat);
                // project to tangent plane
                o = sub(o, kmul(dot(o, n), n));
                o = normalize(o);
                w += 1;
            }
            orientation_field[vi] = o;
        }
    }

    // smooth position field
    println!("Smoothing position field");
    for _it in 0..args.pos_smoothing_iters {
        for vi in 0..nv {
            let o = orientation_field[vi];
            let mut p = pos_field[vi];
            let n = vert_normals[vi];
            let vert = v[vi];
            let mut w = 0;
            for &adj_vi in vv_adj.adj(vi) {
                let adj_vi = adj_vi as usize;
                let p_compat = compat_pos_extrinsic(
                    o,
                    p,
                    n,
                    vert,
                    orientation_field[adj_vi],
                    pos_field[adj_vi],
                    vert_normals[adj_vi],
                    v[adj_vi],
                    args.scale,
                );
                p = divk(add(kmul(w as F, p_compat[0]), p_compat[1]), (w + 1) as F);
                p = sub(p, kmul(dot(n, sub(p, vert)), n));
                w += 1;
            }
            pos_field[vi] = lattice_op(p, o, n, vert, args.scale, RoundMode::Round);
        }
    }

    pos_field
}

fn compat_orientation_extrinsic(o0: [F; 3], n0: [F; 3], o1: [F; 3], n1: [F; 3]) -> [[F; 3]; 2] {
    let a = [o0, cross(n0, o0), o0.map(Neg::neg), cross(o0, n0)];
    let b = [o1, cross(n1, o1)];
    let mut ai = 0;
    let mut bi = 0;
    let mut highest_prod = F::NEG_INFINITY;
    for (i, a) in a.into_iter().enumerate() {
        for (j, b) in b.into_iter().enumerate() {
            let prod = dot(a, b);
            if prod > highest_prod {
                highest_prod = prod;
                ai = i;
                bi = j;
            }
        }
    }
    [a[ai], b[bi]]
}

/*
def compat_position_extrinsic(o0, p0, n0, v0, o1, p1, n1, v1, scale):
    """ Find compatible versions of two representative positions (with specified normals and orientations) """
    t0, t1, middle = np.cross(n0, o0), np.cross(n1, o1), intermediate_pos(v0, n0, v1, n1)
    p0, p1 = lattice_op(p0, o0, n0, middle, scale), lattice_op(p1, o1, n1, middle, scale)
    x = min(all_combinations([0, 1], [0, 1], [0, 1], [0, 1]),
        key = lambda x : np.linalg.norm((p0 + scale * (o0 * x[0] + t0 * x[1])) - (p1 + scale * (o1 * x[2] + t1 * x[3]))))
    result = (p0 + scale * (o0 * x[0] + t0 * x[1]), p1 + scale * (o1 * x[2] + t1 * x[3]))
    return result
*/
fn compat_pos_extrinsic(
    o0: [F; 3],
    p0: [F; 3],
    n0: [F; 3],
    v0: [F; 3],
    o1: [F; 3],
    p1: [F; 3],
    n1: [F; 3],
    v1: [F; 3],
    scale: F,
) -> [[F; 3]; 2] {
    let t0 = cross(n0, o0);
    let t1 = cross(n1, o1);
    let mid = intermediate_pos(v0, n0, v1, n1);
    let p0 = lattice_op(p0, o0, n0, mid, scale, RoundMode::Floor);
    let p1 = lattice_op(p1, o1, n1, mid, scale, RoundMode::Floor);

    let combos: [[bool; 4]; 16] =
        std::array::from_fn(|i| [i / 8 == 0, (i / 4) % 2 == 0, (i / 2) % 2 == 0, i % 2 == 0]);
    fn cond(b: bool, x: [F; 3]) -> [F; 3] {
        if b { x } else { [0.; 3] }
    }
    let (term0, term1, _) = combos
        .into_iter()
        .map(|[a0, b0, a1, b1]| {
            let term0 = add(p0, kmul(scale, add(cond(a0, o0), cond(b0, t0))));
            let term1 = add(p1, kmul(scale, add(cond(a1, o1), cond(b1, t1))));
            (term0, term1, length(sub(term0, term1)))
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    [term0, term1]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    Round,
    Floor,
}

impl RoundMode {
    pub fn apply(self, f: F) -> F {
        match self {
            RoundMode::Round => f.round(),
            RoundMode::Floor => f.floor(),
        }
    }
}

// 4 positional rotational symmetric floor/rounding operation
fn lattice_op(
    pos: [F; 3],
    o: [F; 3],
    n: [F; 3],
    tgt /* vertex */: [F; 3],
    scale: F,
    round_mode: RoundMode,
) -> [F; 3] {
    let t = cross(n, o);
    let d = sub(tgt, pos);
    let od = round_mode.apply(dot(o, d) / scale);
    let td = round_mode.apply(dot(t, d) / scale);
    let delta = kmul(scale, add(kmul(od, o), kmul(td, t)));
    add(pos, delta)
}

/*
def lattice_op(p, o, n, target, scale, op = np.floor):
    """ 4-PoSy lattice floor/rounding operation -- see the paper appendix for details """
    t, d = np.cross(n, o), target - p
    return p + scale * (o * op(np.dot(o, d) / scale) + t * op(np.dot(t, d) / scale))
*/

fn intermediate_pos(p0: [F; 3], n0: [F; 3], p1: [F; 3], n1: [F; 3]) -> [F; 3] {
    let n0p0 = dot(n0, p0);
    let n1p0 = dot(n1, p0);
    let n0p1 = dot(n0, p1);
    let n1p1 = dot(n1, p1);
    let n0n1 = dot(n0, n1);

    let inv_denom = (1. - n0n1 * n0n1).max(1e-8);
    let l0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1));
    let l1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0));
    let fst = kmul(0.5, add(p0, p1));
    let snd = divk(add(kmul(l0, n0), kmul(l1, n1)), inv_denom * 4.);
    sub(fst, snd)
}

/*
def intermediate_pos(p0, n0, p1, n1):
    """ Find an intermediate position between two vertices -- see the paper appendix """
    n0p0, n0p1, n1p0, n1p1, n0n1 = np.dot(n0, p0), np.dot(n0, p1), np.dot(n1, p0), np.dot(n1, p1), np.dot(n0, n1)
    denom = 1.0 / (1.0 - n0n1*n0n1 + 1e-4)
    lambda_0 = 2.0*(n0p1 - n0p0 - n0n1*(n1p0 - n1p1))*denom
    lambda_1 = 2.0*(n1p0 - n1p1 - n0n1*(n0p1 - n0p0))*denom
    return 0.5 * (p0 + p1) - 0.25 * (n0 * lambda_0 + n1 * lambda_1)
    */
