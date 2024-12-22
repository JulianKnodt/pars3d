use super::{cross, dot, edges, normalize, sub, FaceKind, F};
use std::cmp::minmax;
use std::collections::HashMap;

use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdFloat(F);
impl Eq for OrdFloat {}

impl PartialOrd for OrdFloat {
    fn partial_cmp(&self, f: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(f))
    }
}
impl Ord for OrdFloat {
    fn cmp(&self, f: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&f.0)
    }
}

/// Metric used for preference when converting triangles to quads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadPreference {
    /// Prefer constructing pi/2 angles.
    RightAngle,
    /// Prefer constructing quads with opposing angles being the most similar.
    Symmetric,
}

/// Quadrangulate a set of vertices and faces.
/// Epsilon is given in \[0,1\] for both.
pub fn quadrangulate(
    vs: &[[F; 3]],
    faces: &[FaceKind],
    planarity_eps: F,
    angle_eps: F,
    quad_pref: QuadPreference,
) -> (Vec<FaceKind>, Vec<(usize, Option<usize>)>) {
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

        // repeat faces
        if corner_a == corner_b {
            continue;
        }

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
        let new_angle1 = dot(normalize(sub(v2, v3)), normalize(sub(v0, v3)))
            .clamp(-1., 1.)
            .acos();
        let angle_metric = match quad_pref {
            QuadPreference::RightAngle => {
                const HALF_PI: F = std::f64::consts::FRAC_PI_2 as F;
                let delta0 = (new_angle0 - HALF_PI).abs();
                if delta0 > angle_eps {
                    continue;
                }

                let delta1 = (new_angle1 - HALF_PI).abs();
                if delta1 > angle_eps {
                    continue;
                }
                delta0 + delta1
            }
            QuadPreference::Symmetric => {
                let delta = (new_angle0 - new_angle1).abs();
                if delta > angle_eps {
                    continue;
                }
                delta
            }
        };

        // higher is better for this alignment
        // lower is better for angle metric
        merge_heap.push((OrdFloat(-angle_metric), OrdFloat(align), new_quad, [a, b]));
    }

    let mut new_faces = vec![];
    let mut deleted = vec![false; faces.len()];
    let mut merged = vec![];

    let mut snd_heap = BinaryHeap::new();

    while let Some((ang, align, q, [a, b])) = merge_heap.pop() {
        snd_heap.push((align, ang, q, [a, b]));

        while let Some((_, ang, q, [a, b])) = snd_heap.pop() {
            if deleted[a] || deleted[b] {
                continue;
            }
            deleted[a] = true;
            deleted[b] = true;
            new_faces.push(FaceKind::Quad(q));
            merged.push((a, Some(b)));

            while let Some(&(na, nal, nq, nab)) = merge_heap.peek()
                && (na.0 - ang.0).abs() < 1e-3
            {
                merge_heap.pop();
                snd_heap.push((nal, na, nq, nab));
            }
        }
    }

    new_faces.extend(
        deleted
            .iter()
            .enumerate()
            .filter(|&(_, del)| !del)
            .map(|(i, _)| faces[i].clone()),
    );

    merged.extend(
        deleted
            .iter()
            .enumerate()
            .filter(|&(_, del)| !del)
            .map(|(i, _)| (i, None)),
    );

    (new_faces, merged)
}

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
        *self = match *self {
            Boundary(a) if a != v => EdgeKind::Manifold([a, v]),
            Manifold([a, b]) if a != v && b != v => EdgeKind::NonManifold,
            _ => return,
        }
    }
}
