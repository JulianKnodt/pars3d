use super::{cross, dot, edges, normalize, sub, F};
use std::cmp::minmax;
use std::collections::HashMap;

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
