use core::ops::Neg;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use super::coloring::magma;
use super::edge::EdgeKind;
use super::{add, cross, edges, kmul, normalize, sub, F};

/// Returns a coherent vertex coloring for a mesh with joint influences.
pub fn joint_influence_coloring(
    joint_influence: impl Fn(usize) -> ([F; 4], [u16; 4]),
    num_vs: usize,
    total_joints: u16,
) -> impl Iterator<Item = [F; 3]> {
    let joint_colors = (0..total_joints)
        .map(|ji| magma(ji as F / (total_joints - 1) as F))
        .collect::<Vec<_>>();

    (0..num_vs).map(move |vi| {
        let (jw, ji) = joint_influence(vi);
        (0..4)
            .map(|i| kmul(jw[i], joint_colors[ji[i] as usize]))
            .fold([0.; 3], add)
    })
}

/// Normalize scalars defined at each vertex from [-min, max] to [0,1], apply a coloring and
/// visualize isosurfaces at a fixed interval.
/// `isolevel_freq` should be in the range [0, 1). If 0 or neg, will be ignored.
/// Color fn should take [0, 1] to a color. One such function is `coloring::magma`.
pub fn vertex_scalar_coloring(
    scalars: &[F],
    color_fn: impl Fn(F) -> [F; 3],
    isolevel_freq: F,
    isolevel_width: F,
    isolevel_color: [F; 3],
) -> Vec<[F; 3]> {
    assert!(isolevel_freq < 1.);
    if scalars.is_empty() {
        return vec![];
    }

    let min = scalars.iter().copied().min_by(F::total_cmp).unwrap();
    assert!(min.is_finite());

    let max = scalars.iter().copied().max_by(F::total_cmp).unwrap();
    assert!(max.is_finite());

    let mut out = Vec::with_capacity(scalars.len());

    let apply_iso = isolevel_freq > 0.;

    for &s in scalars {
        let new = (s - min) / (max - min);
        let c = color_fn(new);
        if apply_iso && (new % isolevel_freq) < isolevel_width {
            let iso = (new % isolevel_freq) / isolevel_freq;
            assert!((0.0..=1.0).contains(&iso));
            let iso = iso.sqrt();
            let interp = std::array::from_fn(|i| iso * isolevel_color[i] + (1. - iso) * c[i]);
            out.push(interp);
            continue;
        }

        out.push(c);
    }
    out
}

/// Constructs a coloring for each face, based on some classification function.
pub fn face_coloring(face_group: impl Fn(usize) -> usize, num_fs: usize) -> Vec<[F; 3]> {
    // note that `sin` below is a cheap way to make it pseudo random
    (0..num_fs)
        // also add num_fs to make it more random.
        .map(|i| {
            std::array::from_fn(|j| {
                (((j * j + 1) as F * 33.31293) * (face_group(i) + num_fs) as F).sin() * 0.5 + 0.5
            })
        })
        .collect()
}

/// When segmenting faces into groups, emit a black wireframe along boundaries between
/// clusterings.
pub fn face_segmentation_wireframes<'a>(
    fs: impl Fn(usize) -> &'a [usize],
    face_group: impl Fn(usize) -> usize,
    nf: usize,
    vertices: &[[F; 3]],
    width: F,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut edge_face_adj: HashMap<[usize; 2], EdgeKind> = HashMap::new();
    for fi in 0..nf {
        let f = fs(fi);
        for [e0, e1] in edges(f) {
            let e = std::cmp::minmax(e0, e1);
            edge_face_adj
                .entry(e)
                .or_insert_with(EdgeKind::empty)
                .insert(fi);
        }
    }

    let mut edges = vec![];
    'outer: for (&[e0, e1], ek) in edge_face_adj.iter() {
        let eks = ek.as_slice();
        for i in 0..eks.len() {
            let fi = eks[i];
            let fi_group = face_group(fi);
            for &fj in &eks[i + 1..eks.len()] {
                if fi_group != face_group(fj) {
                    // emit a cylinder here
                    edges.push([e0, e1]);
                    continue 'outer;
                }
            }
        }
    }
    opt_raw_edge_vector_visualization(edges.into_iter(), |vi| vertices[vi], |_| [0.; 3], width)
}

/// Uses a fixed set of high contrast colors to color each face group.
/// Face group should be O(1). Outputs a color for each face, which can be applied with
/// Mesh::with_face_coloring. `palette` can be arbitrary, but one example is
/// pars3d::coloring::HIGH_CONTRAST.
pub fn greedy_face_coloring(
    // Fn(Face) -> Group#
    face_group: impl Fn(usize) -> usize,
    // #Faces
    num_fs: usize,
    // Whether two groups are adjacent
    group_adj: impl Fn(usize, usize) -> bool,
    // Palette to use for coloring
    palette: &[[u8; 3]],
) -> Vec<[F; 3]> {
    const fn to_rgbf([r, g, b]: [u8; 3]) -> [F; 3] {
        [r as F / 255., g as F / 255., b as F / 255.]
    }
    let mut uniq_groups = vec![];
    for fi in 0..num_fs {
        let g = face_group(fi);
        if !uniq_groups.contains(&g) {
            uniq_groups.push(g);
        }
    }

    // graph coloring index
    let mut coloring: Vec<usize> = vec![];

    // greedy graph coloring
    for i in 0..uniq_groups.len() {
        let mut nbrs = vec![];
        for j in 0..i {
            if group_adj(uniq_groups[i], uniq_groups[j]) {
                nbrs.push(coloring[j]);
            }
        }
        let color = (0..).filter(|v| !nbrs.contains(v)).next().unwrap();
        coloring.push(color);
    }

    assert_eq!(coloring.len(), uniq_groups.len());

    let n = palette.len();
    (0..num_fs)
        .map(|i| {
            let g = face_group(i);
            let uniq_group_idx = uniq_groups.iter().position(|&v| v == g).unwrap();
            let i = coloring[uniq_group_idx];
            let col = to_rgbf(palette[i % n]);
            // common case
            if i < n {
                return col;
            }

            // dense meshes? Make the colors darker.
            let pow = (i / n) as i32;
            col.map(|v| v.powi(pow))
        })
        .collect()
}

impl super::mesh::Mesh {
    /// Constructs a new mesh with a given face coloring.
    /// Does not retain any information from the original mesh.
    pub fn with_face_coloring(&self, fc: &[[F; 3]]) -> Self {
        use super::FaceKind;
        let mut new_v = vec![];
        let mut new_fs = vec![];
        let mut new_vc = vec![];
        let mut created_pairs = HashMap::new();

        fn to_key(v: [F; 3], c: [F; 3]) -> [super::U; 6] {
            [
                v[0].to_bits(),
                v[1].to_bits(),
                v[2].to_bits(),
                c[0].to_bits(),
                c[1].to_bits(),
                c[2].to_bits(),
            ]
        }

        for (fi, f) in self.f.iter().enumerate() {
            let color = fc[fi];
            let mut new_f = FaceKind::empty();
            for &v in f.as_slice() {
                let key = to_key(self.v[v], color);
                match created_pairs.entry(key) {
                    Entry::Occupied(o) => new_f.insert(*o.get()),
                    Entry::Vacant(empty) => {
                        let new_pos = new_v.len();
                        new_v.push(self.v[v]);
                        new_vc.push(color);
                        empty.insert(new_pos);
                        new_f.insert(new_pos);
                    }
                }
            }

            new_fs.push(new_f);
        }

        Self {
            v: new_v,
            f: new_fs,
            vert_colors: new_vc,

            ..Default::default()
        }
    }
}

/// Given a mesh with a per-edge scalar value in 0-1, output a new triangle mesh with vertex
/// coloring that matches the value for each edge. 1 represents max intensity, 0 represents void.
pub fn edge_value_visualization<'a>(
    fs: impl Fn(usize) -> &'a [usize],
    nf: usize,
    vs: &[[F; 3]],
    edge_value: impl Fn([usize; 2]) -> F,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 3]>) {
    optional_edge_value_visualization(fs, nf, vs, |e| Some(edge_value(e)), [0., 1., 0.])
}

/// Given a mesh with a per-edge scalar value in 0-1, output a new triangle mesh with vertex
/// coloring that matches the value for each edge. 1 represents max intensity, 0 represents void.
pub fn optional_edge_value_visualization<'a>(
    fs: impl Fn(usize) -> &'a [usize],
    nf: usize,
    vs: &[[F; 3]],
    edge_value: impl Fn([usize; 2]) -> Option<F>,
    default_color: [F; 3],
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 3]>) {
    optional_edge_vector_visualization(fs, nf, vs, |e| {
        edge_value(e).map(magma).unwrap_or(default_color)
    })
}
pub fn optional_edge_vector_visualization<'a>(
    fs: impl Fn(usize) -> &'a [usize],
    nf: usize,
    vs: &[[F; 3]],
    edge_value: impl Fn([usize; 2]) -> [F; 3],
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 3]>) {
    let mut new_vs = vec![];
    let mut new_vc = vec![];

    let mut new_fs = vec![];

    for fi in 0..nf {
        let f = fs(fi);
        let centroid = f.iter().map(|&vi| vs[vi]).fold([0.; 3], add);
        let centroid = kmul((f.len() as F).recip(), centroid);

        for [e0, e1] in edges(f) {
            let e0i = new_vs.len();
            new_vs.push(vs[e0]);
            let e1i = new_vs.len();
            new_vs.push(vs[e1]);

            let midpoint = kmul(0.5, add(vs[e0], vs[e1]));
            let midpoint = add(kmul(0.2, centroid), kmul(0.8, midpoint));
            let midi = new_vs.len();
            new_vs.push(midpoint);

            let edge_color = edge_value([e0, e1]);
            new_vc.push(edge_color);
            new_vc.push(edge_color);
            new_vc.push(edge_color);

            new_fs.push([e0i, e1i, midi]);
        }
        assert_eq!(new_vs.len(), new_vc.len());
    }

    (new_vs, new_vc, new_fs)
}

// TODO rename this to wireframe visualization
pub fn opt_raw_edge_visualization(
    edges: impl Iterator<Item = [usize; 2]>,
    vs: impl Fn(usize) -> [F; 3],
    edge_value: impl Fn([usize; 2]) -> Option<F>,
    default_color: [F; 3],
    width: F,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    opt_raw_edge_vector_visualization(
        edges,
        vs,
        |e| edge_value(e).map(magma).unwrap_or(default_color),
        width,
    )
}

/// emit a cylindrical wireframe of a given color for a set of edges
pub fn opt_raw_edge_vector_visualization(
    edges: impl Iterator<Item = [usize; 2]>,
    vs: impl Fn(usize) -> [F; 3],
    edge_value: impl Fn([usize; 2]) -> [F; 3],
    width: F,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut new_vs = vec![];
    let mut new_vc = vec![];
    let mut new_fs = vec![];

    for [e0, e1] in edges {
        let color = edge_value([e0, e1]);
        let e0 = vs(e0);
        let e1 = vs(e1);
        let e_dir = normalize(sub(e1, e0));
        if e_dir.iter().all(|&v| v == 0.) {
            continue;
        }
        let np = non_parallel(e_dir);
        let t = normalize(cross(e_dir, np));
        let b = normalize(cross(e_dir, t));
        let t = kmul(width, t);
        let b = kmul(width, b);
        let q0 = [
            add(e0, t),
            add(e0, b),
            add(e0, t.map(Neg::neg)),
            add(e0, b.map(Neg::neg)),
        ];
        let q1 = [
            add(e1, t),
            add(e1, b),
            add(e1, t.map(Neg::neg)),
            add(e1, b.map(Neg::neg)),
        ];
        let c = new_vs.len();
        new_vs.extend(q0.into_iter());
        new_vs.extend(q1.into_iter());
        for _ in 0..8 {
            new_vc.push(color);
        }
        for i in 0..4 {
            new_fs.push([i, (i + 1) % 4, ((i + 1) % 4) + 4, i + 4].map(|v| v + c));
        }
    }

    (new_vs, new_vc, new_fs)
}

fn non_parallel([x, y, z]: [F; 3]) -> [F; 3] {
    if (x - y).abs() > 1e-3 {
        [y, x, z]
    } else if (x - z).abs() > 1e-3 {
        [z, y, x]
    } else if (y - z).abs() > 1e-3 {
        [x, z, y]
    } else {
        [-x, y, z]
    }
}

#[ignore]
#[test]
fn test_edge_vis() {
    let (vs, vc, fs) = edge_value_visualization(
        |_| &[0, 1, 2, 3],
        1,
        &[[0.; 3], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        |[i, _j]| (i + 1) as F / 4.,
    );
    use super::ply::Ply;
    let vc = vc
        .into_iter()
        .map(|rgb| rgb.map(|v| (v * 255.) as u8))
        .collect::<Vec<_>>();

    let fs = fs.into_iter().map(super::FaceKind::Tri).collect::<Vec<_>>();

    let ply = Ply::new(vs, vc, vec![], vec![], fs);
    use std::fs::File;
    let f = File::create("edge_vis.ply").unwrap();
    ply.write(f).unwrap();
}

#[ignore]
#[test]
fn test_raw_edge_vis() {
    let vs = [[0.; 3], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
    let (vs, vc, fs) = opt_raw_edge_visualization(
        [[0, 1], [1, 2], [2, 3], [3, 0]].into_iter(),
        |vi| vs[vi],
        |[i, _j]| Some((i + 1) as F / 4.),
        [0., 1., 0.],
        1e-2,
    );
    use super::ply::Ply;
    let vc = vc
        .into_iter()
        .map(|rgb| rgb.map(|v| (v * 255.) as u8))
        .collect::<Vec<_>>();

    let fs = fs
        .into_iter()
        .map(super::FaceKind::Quad)
        .collect::<Vec<_>>();

    let ply = Ply::new(vs, vc, vec![], vec![], fs);
    use std::fs::File;
    let f = File::create("raw_edge_vis.ply").unwrap();
    ply.write(f).unwrap();
}
