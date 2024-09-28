use super::{cross, dot, edges, normalize, sub, FaceKind, F};
use std::cmp::minmax;
use std::collections::HashMap;

use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrdFloat(F);
impl Eq for OrdFloat {}

impl Ord for OrdFloat {
    fn cmp(&self, f: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&f.0)
    }
}

pub fn quadrangulate(
    vs: &[[F; 3]],
    faces: &[FaceKind],
    planarity_eps: F,
    angle_eps: F,
) -> Vec<FaceKind> {
    let mut edge_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();

    let mut tri_normals = vec![[0.; 3]; faces.len()];
    for (fi, f) in faces.iter().enumerate() {
        let FaceKind::Tri(t) = f else {
            for [e0, e1] in edges(f.as_slice()) {
                edge_adj.insert(minmax(e0, e1), EdgeKind::NonManifold);
            }
            continue;
        };

        for [e0, e1] in edges(t) {
            edge_adj
                .entry(minmax(e0, e1))
                .and_modify(|v| v.insert(fi))
                .or_insert(EdgeKind::Boundary(fi));
        }
        let [v0, v1, v2] = t.map(|vi| vs[vi]);
        tri_normals[fi] = normalize(cross(sub(v2, v0), sub(v1, v0)));
    }

    let mut merge_heap = BinaryHeap::new();

    for ([e0, e1], ek) in edge_adj.into_iter() {
        let EdgeKind::Manifold([a, b]) = ek else {
            continue;
        };
        let align = dot(tri_normals[a], tri_normals[b]);
        // same normals
        if align < 1. - planarity_eps {
            continue;
        }

        let FaceKind::Tri(mut tri_a) = faces[a] else {
            panic!();
        };
        let Some(corner_a) = tri_a.into_iter().find(|&v| v != e0 && v != e1) else {
            continue;
        };
        let FaceKind::Tri(tri_b) = faces[b] else {
            panic!();
        };
        let Some(corner_b) = tri_b.into_iter().find(|&v| v != e0 && v != e1) else {
            continue;
        };
        assert_ne!(corner_a, corner_b);

        while tri_a[0] != corner_a {
            tri_a.rotate_left(1);
        }

        let new_quad: [usize; 4] = std::array::from_fn(|i| match i {
            0 | 1 => tri_a[i],
            2 => corner_b,
            _ => tri_a[2],
        });
        let [v0, v1, v2, v3] = new_quad.map(|vi| vs[vi]);
        let new_angle0 = dot(normalize(sub(v0, v1)), normalize(sub(v2, v1)))
            .clamp(-1., 1.)
            .acos();
        const HALF_PI: F = std::f64::consts::FRAC_PI_2 as F;
        if (new_angle0 - HALF_PI).abs() > angle_eps {
            continue;
        }

        let new_angle1 = dot(normalize(sub(v2, v3)), normalize(sub(v0, v3)))
            .clamp(-1., 1.)
            .acos();
        if (new_angle1 - HALF_PI).abs() > angle_eps {
            continue;
        }

        use std::cmp::Reverse;
        merge_heap.push((
            Reverse(OrdFloat(
                (new_angle0 - HALF_PI).abs() + (new_angle1 - HALF_PI).abs(),
            )),
            // higher is better for this one
            OrdFloat(align),
            new_quad,
            [a, b],
        ));
    }

    let mut new_faces = vec![];
    let mut deleted = vec![false; faces.len()];

    let mut snd_heap = BinaryHeap::new();

    while let Some((ang, align, q, [a,b])) = merge_heap.pop() {
        snd_heap.push((align, ang, q, [a,b]));

        while let Some((_, ang, q, [a,b])) = snd_heap.pop() {
            if deleted[a] || deleted[b] {
                continue;
            }
            while let Some(&(na, nal, nq, nab)) = merge_heap.peek() &&
              ((na.0).0 - (ang.0).0).abs() < 1e-6 {
              merge_heap.pop();
              snd_heap.push((nal, na, nq, nab));
            }
            deleted[a] = true;
            deleted[b] = true;
            new_faces.push(FaceKind::Quad(q));
        }
    }

    new_faces.extend(
        deleted
            .into_iter()
            .enumerate()
            .filter(|&(_, del)| !del)
            .map(|(i, _)| faces[i].clone()),
    );

    new_faces
}

/*
/// Attempt to convert a triangle mesh to a quad mesh.
/// Converts flat edges, where the new quad has as close to 90 degree angles as possible.
pub fn tri_to_quad(
    vs: &[[F; 3]],
    tris: &[[usize; 3]],
    planarity_eps: F,
    angle_eps: F,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut quads = vec![];
    let mut deleted = vec![false; tris.len()];

    let mut edge_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();

    let mut tri_normals = vec![[0.; 3]; tris.len()];
    for (ti, t) in tris.iter().enumerate() {
        for [e0, e1] in edges(t) {
            edge_adj
                .entry(minmax(e0, e1))
                .and_modify(|v| v.insert(ti))
                .or_insert(EdgeKind::Boundary(ti));
        }
        let [v0, v1, v2] = t.map(|vi| vs[vi]);
        tri_normals[ti] = normalize(cross(sub(v2, v0), sub(v1, v0)));
    }

    for ([e0, e1], ek) in edge_adj.into_iter() {
        let EdgeKind::Manifold([a, b]) = ek else {
            continue;
        };
        if deleted[a] || deleted[b] {
            continue;
        }
        // same normals
        if dot(tri_normals[a], tri_normals[b]) < 1. - planarity_eps {
            continue;
        }
        let Some(corner_a) = tris[a].into_iter().find(|&v| v != e0 && v != e1) else {
            continue;
        };
        let Some(corner_b) = tris[b].into_iter().find(|&v| v != e0 && v != e1) else {
            continue;
        };

        let mut tri_a = tris[a];
        while !tri_a[0] == corner_a {
            tri_a.rotate_left(1);
        }

        let new_quad: [usize; 4] = std::array::from_fn(|i| {
            if i == 0 || i == 1 {
                tri_a[i]
            } else if i == 2 {
                corner_b
            } else {
                tri_a[2]
            }
        });
        let [v0, v1, v2, v3] = new_quad.map(|vi| vs[vi]);
        let new_angle = dot(normalize(sub(v0, v1)), normalize(sub(v2, v2)))
            .clamp(-1., 1.)
            .acos();
        if (new_angle - std::f64::consts::FRAC_PI_2 as F).abs() < angle_eps {
            continue;
        }

        let new_angle = dot(normalize(sub(v2, v3)), normalize(sub(v0, v3)))
            .clamp(-1., 1.)
            .acos();
        if (new_angle - std::f64::consts::FRAC_PI_2 as F).abs() < angle_eps {
            continue;
        }

        deleted[a] = true;
        deleted[b] = true;
        quads.push(new_quad);
    }

    let tris = deleted
        .into_iter()
        .enumerate()
        .filter(|&(_, del)| !del)
        .map(|(i, _)| tris[i])
        .collect::<Vec<[usize; 3]>>();

    (quads, tris)
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeKind {
    Boundary(usize),
    Manifold([usize; 2]),
    // We do not need to track any data for non-manifold edges here.
    NonManifold,
}

impl EdgeKind {
    fn insert(&mut self, v: usize) {
        use EdgeKind::*;
        *self = match self {
            &mut Boundary(a) if a != v => EdgeKind::Manifold([a, v]),
            &mut Manifold([a, b]) if a != v && b != v => EdgeKind::NonManifold,
            _ => return,
        }
    }
}
