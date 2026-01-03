use crate::{F, add, cross_2d, dist, dot, edges, kmul, length, length_sq, sub, tri_area_2d};

use std::array::from_fn;

use std::collections::BTreeSet;

/// Delaunay triangulation of 2D points
pub fn bowyer_watson_2d(ps: &[[F; 2]]) -> impl Iterator<Item = [usize; 3]> {
    const N: usize = 2;
    let mut min = [F::INFINITY; 2];
    let mut max = [F::NEG_INFINITY; 2];
    // Cubic cost (TODO optimize?) it is easily parallelizable though
    for i in 0..ps.len() {
        for j in (i + 1)..ps.len() {
            for k in (j + 1)..ps.len() {
                let (c, r) = circumcircle_2d([i, j, k].map(|vi| ps[vi]));
                for i in 0..2 {
                    min[i] = min[i].min(c[i] - r);
                    max[i] = max[i].max(c[i] + r);
                }
            }
        }
    }

    let w = (0..N)
        .map(|i| max[i] - min[i])
        .max_by(F::total_cmp)
        .unwrap();

    let super_simplex = [
        [min[0] - 0.5 * w, min[1]],
        [max[0] + 0.5 * w, min[1]],
        [(min[0] + max[0]) / 2., min[1] + 2. * w],
    ];

    let markers: [usize; N + 1] = from_fn(|i| usize::MAX - i);

    let mut simps = BTreeSet::new();
    simps.insert(markers);

    let get_v = |i: usize| {
        let t = usize::MAX - i;
        if t < N + 1 { super_simplex[t] } else { ps[i] }
    };

    // TODO bad tris can be omitted by swapping to the end and keeping a count
    let mut bad_tris = vec![];
    let mut polys: BTreeSet<[usize; 2]> = BTreeSet::new();
    for (pi, p) in ps.iter().enumerate() {
        bad_tris.clear();
        for &s in simps.iter() {
            let tri = s.map(get_v);
            let cc = circumcircle_2d(tri);
            if circle_contains(cc, *p) {
                bad_tris.push(s);
            }
        }

        polys.clear();
        for bt in bad_tris.drain(..) {
            for [e0, e1] in edges(&bt) {
                use std::collections::btree_set::Entry;
                match polys.entry(std::cmp::minmax(e0, e1)) {
                    Entry::Occupied(o) => {
                        o.remove();
                    }
                    Entry::Vacant(v) => v.insert(),
                }
            }
            let ok = simps.remove(&bt);
            assert!(ok);
        }

        for &[e0, e1] in polys.iter() {
            simps.insert([e0, e1, pi]);
        }
    }

    // clear super-simplex
    simps.retain(|s| s.iter().all(|&v| v < usize::MAX - N));

    simps.into_iter().map(|mut s| {
        if tri_area_2d(s.map(|vi| ps[vi])) < 0. {
            s.reverse();
        }
        s
    })
}

pub fn bowyer_watson_3d(ps: &[[F; 3]]) -> impl Iterator<Item = [usize; 4]> {
    const N: usize = 3;
    let mut min = [F::INFINITY; 3];
    let mut max = [F::NEG_INFINITY; 3];
    // Cubic cost (TODO optimize?) it is easily parallelizable though
    for i in 0..ps.len() {
        for j in (i + 1)..ps.len() {
            for k in (j + 1)..ps.len() {
                for l in (k + 1)..ps.len() {
                    let (c, r) = circumsphere_tet([i, j, k, l].map(|vi| ps[vi]));
                    for i in 0..N {
                        min[i] = min[i].min(c[i] - r);
                        max[i] = max[i].max(c[i] + r);
                    }
                }
            }
        }
    }

    let w = (0..N)
        .map(|i| max[i] - min[i])
        .max_by(F::total_cmp)
        .unwrap();

    // not sure if this is tight, but should enclose all points.
    let super_simplex = [
        [min[0] - 4. * w, min[1], min[2]],
        [max[0] + 4. * w, min[1], min[2]],
        [(min[0] + max[0]) / 2., min[1] + 7. * w, min[2]],
        [(min[0] + max[0]) / 2., min[1], min[2] + 7. * w],
    ];

    let markers: [usize; N + 1] = from_fn(|i| usize::MAX - i);

    let tet_tris = |[a, b, c, d]: [usize; 4]| [[a, b, c], [a, b, d], [a, c, d], [b, c, d]];

    let mut simps: BTreeSet<[usize; N + 1]> = BTreeSet::new();
    simps.insert(markers);

    let get_v = |i: usize| {
        let t = usize::MAX - i;
        if t < N + 1 { super_simplex[t] } else { ps[i] }
    };

    let mut bad_tets = vec![];
    let mut polys: BTreeSet<[usize; 3]> = BTreeSet::new();
    for (pi, p) in ps.iter().enumerate() {
        bad_tets.clear();
        for &s in simps.iter() {
            let tet = s.map(get_v);
            let cc = circumsphere_tet(tet);
            if circle_contains(cc, *p) {
                bad_tets.push(s);
            }
        }

        polys.clear();
        for bt in bad_tets.drain(..) {
            for mut tri in tet_tris(bt) {
                tri.sort_unstable();
                use std::collections::btree_set::Entry;
                match polys.entry(tri) {
                    Entry::Occupied(o) => {
                        o.remove();
                    }
                    Entry::Vacant(v) => v.insert(),
                }
            }
            let ok = simps.remove(&bt);
            assert!(ok);
        }

        for &[v0, v1, v2] in polys.iter() {
            simps.insert([v0, v1, v2, pi]);
        }
    }

    // clear super-simplex
    simps.retain(|s| s.iter().all(|&v| v < usize::MAX - N));

    simps.into_iter().map(|mut s| {
        if crate::signed_tet_vol(s.map(|vi| ps[vi])) < 0. {
            s.reverse();
        }
        s
    })
}

#[test]
fn test_bowyer_watson_2d() {
    let ps = [[0.; 2], [1., 0.], [1., 1.], [0., 1.], [0.5; 2]];
    let s = bowyer_watson_2d(&ps);
    let p = crate::ply::Ply::new(
        ps.iter().map(|&[x, y]| [x, y, 0.]).collect(),
        vec![],
        vec![],
        vec![],
        s.map(crate::FaceKind::Tri).collect(),
    );
    let f = std::fs::File::create("bowyer_watson_2d.ply").unwrap();
    p.write(f, Default::default()).unwrap();
}

#[test]
fn test_bowyer_watson_3d() {
    let mut ps = vec![];
    const N: usize = 9;
    for i in 0..N {
        let t = i as F / N as F;
        ps.push([
            (t * 33.23 + 0.3).sin(),
            (t * 11.57 + 0.7).sin(),
            (t * 19.44 + 0.58).cos(),
        ]);
    }
    let s = bowyer_watson_3d(&ps);
    let p = crate::ply::Ply::new(
        ps.clone(),
        vec![],
        vec![],
        vec![],
        s.flat_map(|[a, b, c, d]| [[a, b, c], [a, c, d], [a, b, d], [b, c, d]])
            .map(crate::FaceKind::Tri)
            .collect(),
    );
    let f = std::fs::File::create("bowyer_watson_3d.ply").unwrap();
    p.write(f, Default::default()).unwrap();
}

// https://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
/*
fn circumsphere_3d([a, b, c]: [[F; 3]; 3]) -> ([F; 3], F) {
    // flipped names are correct TODO fix them
    let ac = sub(c, a);
    let ab = sub(b, a);
    let ab_x_ac = cross(ab, ac);

    // this is the vector from a TO the circumsphere center
    let t0 = kmul(length_sq(ac), cross(ab_x_ac, ab));
    let t1 = kmul(length_sq(ab), cross(ac, ab_x_ac));
    let to_cc = divk(add(t0, t1), 2. * length_sq(ab_x_ac));
    let rad = length(to_cc);

    // The 3 space coords of the circumsphere center then:
    (add(a, to_cc), rad)
}
*/

fn circumsphere_tet([a, b, c, d]: [[F; 3]; 4]) -> ([F; 3], F) {
    let a_mat = [sub(b, a), sub(c, a), sub(d, a)];
    let a_l2 = length_sq(a);
    let b_vec = kmul(
        0.5,
        [
            length_sq(b) - a_l2,
            length_sq(c) - a_l2,
            length_sq(d) - a_l2,
        ],
    );
    let a_mat = crate::inverse(a_mat);
    let cc = crate::matvecmul3(a_mat, b_vec);

    debug_assert!(
        (dist(cc, a) - dist(cc, b)).abs() < 1e-1,
        "{:?}",
        [a, b, c, d].map(|v| dist(cc, v))
    );
    (cc, dist(cc, a))
}

fn circumdelta([a, b, c]: [[F; 2]; 3]) -> [F; 2] {
    let ba = sub(b, a);
    let ca = sub(c, a);

    let ba2 = dot(ba, ba);
    let ca2 = dot(ca, ca);
    let d = 0.5 / cross_2d(ba, ca);

    let x = ca[1] * ba2 - ba[1] * ca2;
    let y = ba[0] * ca2 - ca[0] * ba2;
    kmul(d, [x, y])
}

fn circumcircle_2d(abc: [[F; 2]; 3]) -> ([F; 2], F) {
    let cd = circumdelta(abc);

    (add(abc[0], cd), length(cd))
}

fn circle_contains<const N: usize>((c, r): ([F; N], F), p: [F; N]) -> bool {
    dist(c, p) < r
}
