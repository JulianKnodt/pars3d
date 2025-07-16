// Instant Meshes
/*

Based off of Wenzel Jakob's Instant Meshes, Python Version
*/

use crate::aabb::AABB;
use crate::{F, FaceKind, add, cross, dist, dist_sq, divk, dot, kmul, normalize, sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Args {
    pub orientation_smoothing_iters: usize,
    pub pos_smoothing_iters: usize,
    pub scale: F,
    pub dissolve_edges: bool,
    /// Force vertices to remain on boundaries of the mesh as much as possible.
    pub dirichlet_boundary: bool,
    // TODO von-neumann boundary condition for orientation field
}
impl Default for Args {
    fn default() -> Self {
        Args {
            orientation_smoothing_iters: 100,
            pos_smoothing_iters: 100,
            scale: 0.01,
            dissolve_edges: false,
            dirichlet_boundary: true,
        }
    }
}

pub fn instant_mesh(
    v: &[[F; 3]],
    f: &[FaceKind],
    vn: &[[F; 3]],
    field: &mut [[F; 3]],
    mut rand: impl FnMut() -> F,

    args: &Args,
) -> (Vec<[F; 3]>, Vec<FaceKind>) {
    if v.is_empty() || f.is_empty() {
        return (vec![], vec![]);
    }
    assert_eq!(field.len(), v.len());
    assert_eq!(vn.len(), v.len());
    let nv = v.len();

    let aabb = AABB::from_slice(v);
    let mut orient_field = vec![[0.; 3]; nv];
    let mut pos_field = vec![[0.; 3]; nv];

    for (i, &vec) in field.iter().enumerate() {
        orient_field[i] = normalize(vec);
    }

    for p in &mut pos_field {
        let t = rand().fract();
        *p = add(kmul(1. - t, aabb.min), kmul(t, aabb.max));
    }

    let vv_adj = crate::adjacency::vertex_vertex_adj(nv, f);

    // smooth orientation field
    println!(
        "Smoothing orientation field for {} iters",
        args.orientation_smoothing_iters
    );
    #[allow(unused_mut)]
    let mut order = (0..nv).collect::<Vec<_>>();
    #[cfg(feature = "rand")]
    let mut rng = rand::rng();
    #[cfg(feature = "rand")]
    use rand::prelude::SliceRandom;

    for _it in 0..args.orientation_smoothing_iters {
        #[cfg(feature = "rand")]
        order.shuffle(&mut rng);
        for &vi in &order {
            let mut o = orient_field[vi];
            let n = vn[vi];
            let mut w = 0;
            for &adj_vi in vv_adj.adj(vi) {
                let adj_vi = adj_vi as usize;
                let [o_base, compat] =
                    compat_orientation_extrinsic_4(o, n, orient_field[adj_vi], vn[adj_vi]);
                o = add(kmul(w as F, o_base), compat);
                // project to tangent plane
                o = sub(o, kmul(dot(o, n), n));
                o = normalize(o);
                w += 1;
            }
            orient_field[vi] = o;
        }
    }
    // make const
    let orient_field = orient_field;

    // smooth position field
    println!(
        "Smoothing position field for {} iters",
        args.pos_smoothing_iters
    );
    for _it in 0..args.pos_smoothing_iters {
        #[cfg(feature = "rand")]
        order.shuffle(&mut rng);
        for &vi in &order {
            let o = orient_field[vi];
            let mut p = pos_field[vi];
            let n = vn[vi];
            let vert = v[vi];
            let mut w = 0;
            for &adj_vi in vv_adj.adj(vi) {
                let adj_vi = adj_vi as usize;
                let (p_compat, _err) = compat_pos_extrinsic_4(
                    o,
                    p,
                    n,
                    vert,
                    //
                    orient_field[adj_vi],
                    pos_field[adj_vi],
                    vn[adj_vi],
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

    // output verts
    let mut to_del = vec![];
    let mut can_dissolve = vec![];
    let mut nbr_edges = vec![];

    for f in f {
        for [vi0, vi1] in f.edges() {
            let oc = compat_orientation_extrinsic_4(
                orient_field[vi0],
                vn[vi0],
                orient_field[vi1],
                vn[vi1],
            );

            let (pc, _err) = compat_pos_extrinsic_4_idx(
                oc[0],
                pos_field[vi0],
                vn[vi0],
                v[vi0],
                //
                oc[1],
                pos_field[vi1],
                vn[vi1],
                v[vi1],
                //
                args.scale,
            );

            let abs_diff = subi(pc[0], pc[1]).map(i32::abs);
            if abs_diff.iter().any(|&v| v > 1) {
                continue;
            }
            let e = std::cmp::minmax(vi0, vi1);
            if args.dissolve_edges && abs_diff == [1, 1] {
                can_dissolve.push(e);
            }
            let abs_diff_sum = abs_diff.iter().sum::<i32>();
            let dst = match abs_diff_sum {
                0 => &mut to_del,
                1 => &mut nbr_edges,
                _ => &mut can_dissolve,
            };
            dst.push(e);
        }
    }

    let mut out_faces = f.to_vec();

    // collapse adjacent vertices
    // (TODO use barycentric area weighting?)
    let mut c = super::collapsible::Collapsible::new_with(pos_field.len(), |vi| (pos_field[vi], 1));
    for f in &out_faces {
        c.add_face(f.as_slice());
    }

    for vis in to_del {
        let [vi0, vi1] = vis.map(|vi| c.get_new_vertex(vi));
        if vi0 == vi1 {
            continue;
        }
        // TODO use a better way to merge vertices here
        c.merge(vi0, vi1, |&(p0, c0), &(p1, c1)| (add(p0, p1), c0 + c1));
    }

    for (vi, &(new_pos, cnt)) in c.vertices() {
        pos_field[vi] = divk(new_pos, cnt as F);
    }

    for f in &mut out_faces {
        f.remap(|vi| c.get_new_vertex(vi));
    }
    out_faces.retain_mut(|f| !f.canonicalize());

    for e in &mut can_dissolve {
        *e = std::cmp::minmax(c.get_new_vertex(e[0]), c.get_new_vertex(e[1]));
    }
    can_dissolve.retain(|[e0, e1]| e0 != e1);
    can_dissolve.sort();
    can_dissolve.dedup();

    for e in &mut nbr_edges {
        *e = std::cmp::minmax(c.get_new_vertex(e[0]), c.get_new_vertex(e[1]));
    }

    can_dissolve.sort_by(|&[a0, a1], &[b0, b1]| {
        let a_dist = dist(pos_field[a0], pos_field[a1]);
        let b_dist = dist(pos_field[b0], pos_field[b1]);
        a_dist.partial_cmp(&b_dist).unwrap().reverse()
    });

    /*
    let wf = crate::visualization::colored_wireframe(
        can_dissolve.iter().copied(),
        |vi| pos_field[vi],
        |[_, _]| [1., 0., 0.],
        0.001,
    );
    let out_tmp = crate::visualization::wireframe_to_mesh(wf);
    let tmp_scene = out_tmp.into_scene();
    crate::save("tmp.ply", &tmp_scene).unwrap();
    */

    // dissolve edges
    crate::tri_to_quad::dissolve_edges(&mut out_faces, &can_dissolve);

    let _num_del = crate::geom_processing::delete_non_manifold_duplicates(&mut out_faces);

    (pos_field, out_faces)
}

fn subi<const N: usize>(a: [i32; N], b: [i32; N]) -> [i32; N] {
    std::array::from_fn(|i| a[i] - b[i])
}

/*
fn compat_orientation_extrinsic_2(o0: [F; 3], _n0: [F; 3], o1: [F; 3], _n1: [F; 3]) -> [[F; 3]; 2] {
    [o0, kmul(dot(o0, o1).signum(), o1)]
}
*/

fn compat_orientation_extrinsic_4(o0: [F; 3], n0: [F; 3], o1: [F; 3], n1: [F; 3]) -> [[F; 3]; 2] {
    let a = [o0, cross(n0, o0)];
    let b = [o1, cross(n1, o1)];
    let mut ai = 0;
    let mut bi = 0;
    let mut highest_prod = F::NEG_INFINITY;
    for (i, a) in a.into_iter().enumerate() {
        for (j, b) in b.into_iter().enumerate() {
            let prod = dot(a, b).abs();
            if prod > highest_prod {
                highest_prod = prod;
                ai = i;
                bi = j;
            }
        }
    }
    let dp = dot(a[ai], b[bi]);
    [a[ai], kmul(dp.signum(), b[bi])]
}

fn compat_pos_extrinsic_4(
    o0: [F; 3],
    p0: [F; 3],
    n0: [F; 3],
    v0: [F; 3],

    o1: [F; 3],
    p1: [F; 3],
    n1: [F; 3],
    v1: [F; 3],
    scale: F,
) -> ([[F; 3]; 2], F) {
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
    let (term0, term1, err) = combos
        .into_iter()
        .map(|[a0, b0, a1, b1]| {
            let term0 = add(p0, kmul(scale, add(cond(a0, o0), cond(b0, t0))));
            let term1 = add(p1, kmul(scale, add(cond(a1, o1), cond(b1, t1))));
            (term0, term1, dist_sq(term0, term1))
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    ([term0, term1], err)
}

fn pos_floor_4_idx(pos: [F; 3], orient: [F; 3], n: [F; 3], v: [F; 3], scale: F) -> [i32; 2] {
    let t = cross(n, orient);
    let d = sub(v, pos);
    [dot(orient, d) / scale, dot(t, d) / scale].map(|v| v.floor() as i32)
}

fn compat_pos_extrinsic_4_idx(
    orient0: [F; 3],
    pos0: [F; 3],
    n0: [F; 3],
    v0: [F; 3],
    //
    orient1: [F; 3],
    pos1: [F; 3],
    n1: [F; 3],
    v1: [F; 3],
    //
    scale: F,
) -> ([[i32; 2]; 2], F) {
    let t0 = cross(n0, orient0);
    let t1 = cross(n1, orient1);
    let mid = intermediate_pos(v0, n0, v1, n1);
    let o0p = pos_floor_4_idx(pos0, orient0, n0, mid, scale);
    let o1p = pos_floor_4_idx(pos1, orient1, n1, mid, scale);

    let combos: [[bool; 4]; 16] =
        std::array::from_fn(|i| [i / 8 == 0, (i / 4) % 2 == 0, (i / 2) % 2 == 0, i % 2 == 0]);
    fn cond(b: bool) -> i32 {
        if b { 1 } else { 0 }
    }
    let ([a0, b0, a1, b1], err) = combos
        .into_iter()
        .map(|c @ [a0, b0, a1, b1]| {
            let mk_term = |p, or, t, a, b, op: [i32; 2]| {
                let v = add(
                    kmul((cond(a) + op[0]) as F, or),
                    kmul((cond(b) + op[1]) as F, t),
                );
                let v = kmul(scale, v);
                add(p, v)
            };
            let term0 = mk_term(pos0, orient0, t0, a0, b0, o0p);
            let term1 = mk_term(pos1, orient1, t1, a1, b1, o1p);
            (c, dist_sq(term0, term1))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let terms = [
        [cond(a0) + o0p[0], cond(b0) + o0p[1]],
        [cond(a1) + o1p[0], cond(b1) + o1p[1]],
    ];
    (terms, err)
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
