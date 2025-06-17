use crate::edge::EdgeKind;
use crate::{add, kmul, F};
use std::cmp::minmax;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub enum BarycentricRepr {
    /// This vertex corresponds to a single vertex
    Vertex(usize),
    /// This vertex corresponds to interpolation along an edge
    Edge([usize; 2], [F; 2]),
    /// This vertex corresponds to an interpolation on along a tri
    Tri([usize; 3], [F; 3]),
}

impl BarycentricRepr {
    pub fn from_triple(a: [usize; 3], ws: [F; 3]) -> Self {
        match (ws, a) {
            ([v, 0., 0.], [vi, _, _]) | ([0., v, 0.], [_, vi, _]) | ([0., 0., v], [_, _, vi]) => {
                assert_eq!(v, 1.);
                BarycentricRepr::Vertex(vi)
            }

            ([a, b, 0.], [ai, bi, _]) | ([0., a, b], [_, ai, bi]) | ([b, 0., a], [bi, _, ai]) => {
                let s = a + b;
                assert_ne!(s, 0.);
                BarycentricRepr::Edge([ai, bi], [a / s, b / s])
            }
            ([a, b, c], vis) => {
                let s = a + b + c;
                BarycentricRepr::Tri(vis, [a, b, c].map(|v| v / s))
            }
        }
    }
    pub fn to_triple(self) -> ([usize; 3], [F; 3]) {
        match self {
            Self::Vertex(vi) => ([vi, usize::MAX, usize::MAX], [1., 0., 0.]),
            Self::Edge([ai, bi], [a, b]) => ([ai, bi, usize::MAX], [a, b, 0.]),
            Self::Tri([ai, bi, ci], [a, b, c]) => ([ai, bi, ci], [a, b, c]),
        }
    }
    /// Given a set of values, computes the interpolated value for this barycentric
    /// representation.
    pub fn eval<const N: usize>(self, vs: &[[F; N]]) -> [F; N] {
        let (idxs, ws) = self.to_triple();
        let mut out = [0.; N];
        for i in 0..3 {
            if idxs[i] == usize::MAX {
                continue;
            }
            out = add(out, kmul(ws[i], vs[idxs[i]]));
        }
        out
    }
}

fn add_triples(
    a_idxs: [usize; 3],
    a_ws: [F; 3],
    b_idxs: [usize; 3],
    b_ws: [F; 3],
) -> ([usize; 3], [F; 3]) {
    let mut out_idxs = [usize::MAX; 3];
    let mut out_ws = [0.; 3];
    for i in 0..3 {
        if b_ws[i] == 0. {
            continue;
        }
        let out_slot = out_idxs
            .iter()
            .position(|&s| s == a_idxs[i] || s == usize::MAX)
            .unwrap();
        out_idxs[out_slot] = a_idxs[i];
        out_ws[out_slot] += a_ws[i];
    }
    for i in 0..3 {
        if b_ws[i] == 0. {
            continue;
        }
        let out_slot = out_idxs
            .iter()
            .position(|&s| s == b_idxs[i] || s == usize::MAX)
            .unwrap();
        out_idxs[out_slot] = b_idxs[i];
        out_ws[out_slot] += b_ws[i];
    }
    (out_idxs, out_ws)
}

/// For a given set of vertices, triangles, and vertex colors, compute the loop subdivided version
/// of the input mesh. Note that the output vertices will be in barycentric form, and extracted
/// values such as position, uv, and color can be computed by calling `eval` for each one.
/// To get the face index of the original triangle that corresponds to a new triangle call
/// `original_tri_index(new_fi, num_subdivs)`.
pub fn loop_subdivision(ts: &[[usize; 3]]) -> (Vec<BarycentricRepr>, Vec<[usize; 3]>) {
    let nv = ts
        .iter()
        .flat_map(|t| t.into_iter().copied().map(|vi| vi + 1))
        .max()
        .unwrap_or(0);
    let mut out_vs = vec![];
    out_vs.extend((0..nv).map(|vi| BarycentricRepr::Vertex(vi)));
    let mut out_ts = vec![];

    let mut edge_tri_map = BTreeMap::new();

    for (ti, &t) in ts.iter().enumerate() {
        for e in tri_edges_ord(t) {
            edge_tri_map
                .entry(e)
                .or_insert_with(EdgeKind::empty)
                .insert(ti);
        }
    }

    let mut counter = out_vs.len();
    let mut edge_vi_map: BTreeMap<[usize; 2], _> = BTreeMap::new();

    // for each edge, insert a new vertex
    for &e @ [e0, e1] in edge_tri_map.keys() {
        edge_vi_map.insert(e, counter);
        counter += 1;
        out_vs.push(BarycentricRepr::Edge([e0, e1], [0.5; 2]));
    }

    for &t in ts.iter() {
        out_ts.push(tri_edges_ord(t).map(|e| edge_vi_map[&e]));

        for [pi, vi, ni] in tri_incident_edges(t) {
            out_ts.push([
                vi,
                edge_vi_map[&minmax(vi, ni)],
                edge_vi_map[&minmax(vi, pi)],
            ]);
        }
    }

    (out_vs, out_ts)
}

/// Returns the original tri index that corresponds to this triangle.
/// Assumes the triangles were not shuffled, and that "subdivision" refers to the
/// `loop_subdivision` function.
pub fn original_tri_index(fi: usize, num_subdivs: usize) -> usize {
    (0..num_subdivs).fold(fi, |acc, _| acc / 4)
}

/// Compose two sets of barycentric coordinates of subdivisions
/// The first argument should be the barycentric coordinates of the subdivided mesh with
/// vertices of base. The output iterator has the same number of elements as subdiv.
/// When subdividing multiple times, it should be:
/// compose(N, compose(..., compose(2, compose(1, 0))))
pub fn compose_barycentric_repr<'a: 'c, 'b: 'c, 'c>(
    subdiv: &'a [BarycentricRepr],
    base: &'b [BarycentricRepr],
) -> impl Iterator<Item = BarycentricRepr> + 'c {
    assert!(subdiv.len() > base.len());

    subdiv.iter().map(|br| {
        let (idxs, ws) = br.to_triple();
        let (new_idxs, new_ws) = (0..3).fold(([usize::MAX; 3], [0.; 3]), |(acc_i, acc_w), i| {
            if ws[i] == 0. {
                return (acc_i, acc_w);
            }
            assert_ne!(idxs[i], usize::MAX);
            let (n_i, n_w) = base[idxs[i]].to_triple();
            add_triples(acc_i, acc_w, n_i, kmul(ws[i], n_w))
        });
        BarycentricRepr::from_triple(new_idxs, new_ws)
    })
}

// TODO make a lazy version of above, which doesn't need to make intermediate representations

#[test]
fn test_loop_subdiv() {
    let tri = vec![[0, 1, 2]];
    let pos = vec![[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]];
    let rgb = vec![[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]];
    let (barys0, tris0) = loop_subdivision(tri.as_slice());
    assert_eq!(barys0.len(), 6);
    let (barys1, tris1) = loop_subdivision(tris0.as_slice());
    assert_eq!(barys1.len(), 15);
    assert_eq!(tris1.len(), 16);

    let new_pos = compose_barycentric_repr(&barys1, &barys0)
        .map(|b| b.eval(&pos))
        .collect::<Vec<_>>();
    let new_rgb = compose_barycentric_repr(&barys1, &barys0)
        .map(|b| b.eval(&rgb))
        .collect::<Vec<_>>();
    let m = crate::Mesh {
        /*
        v: pos,
        f: tri
            .into_iter()
            .map(crate::FaceKind::Tri)
            .collect::<Vec<_>>(),
        vert_colors: rgb,
        */
        v: new_pos,
        f: tris1
            .into_iter()
            .map(crate::FaceKind::Tri)
            .collect::<Vec<_>>(),
        vert_colors: new_rgb,

        ..Default::default()
    };
    let scene = m.into_scene();
    crate::save("test_loop_subdiv.ply", &scene).expect("Failed to save scene");
}

fn tri_edges_ord([vi0, vi1, vi2]: [usize; 3]) -> [[usize; 2]; 3] {
    [minmax(vi0, vi1), minmax(vi1, vi2), minmax(vi0, vi2)]
}

fn tri_incident_edges([vi0, vi1, vi2]: [usize; 3]) -> [[usize; 3]; 3] {
    [
        [vi2, vi0, vi1], //
        [vi0, vi1, vi2],
        [vi1, vi2, vi0],
    ]
}

/*
fn hadamard<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] * b[i])
}
*/
