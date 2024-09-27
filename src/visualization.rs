use super::coloring::magma;
use super::{add, cross, edges, kmul, normalize, sub, F};
use core::ops::Neg;

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

            let edge_color = edge_value([e0, e1]).map(magma).unwrap_or(default_color);
            new_vc.push(edge_color);
            new_vc.push(edge_color);
            new_vc.push(edge_color);

            new_fs.push([e0i, e1i, midi]);
        }
        assert_eq!(new_vs.len(), new_vc.len());
    }

    (new_vs, new_vc, new_fs)
}

pub fn opt_raw_edge_visualization(
    edges: impl Iterator<Item = [usize; 2]>,
    vs: &[[F; 3]],
    edge_value: impl Fn([usize; 2]) -> Option<F>,
    default_color: [F; 3],
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut new_vs = vec![];
    let mut new_vc = vec![];
    let mut new_fs = vec![];

    for [e0, e1] in edges {
        let color = edge_value([e0, e1]).map(magma).unwrap_or(default_color);
        let e0 = vs[e0];
        let e1 = vs[e1];
        let e_dir = normalize(sub(e1, e0));
        if e_dir.iter().all(|&v| v == 0.) {
            continue;
        }
        let np = non_parallel(e_dir);
        let t = normalize(cross(e_dir, np));
        let b = normalize(cross(e_dir, t));
        let t = kmul(0.01, t);
        let b = kmul(0.01, b);
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

    let fs = fs
        .into_iter()
        .map(|tri| super::FaceKind::Tri(tri))
        .collect::<Vec<_>>();

    let ply = Ply::new(vs, vc, fs);
    use std::fs::File;
    let f = File::create("edge_vis.ply").unwrap();
    ply.write(f).unwrap();
}

#[ignore]
#[test]
fn test_raw_edge_vis() {
    let (vs, vc, fs) = opt_raw_edge_visualization(
        [[0, 1], [1, 2], [2, 3], [3, 0]].into_iter(),
        &[[0.; 3], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        |[i, _j]| Some((i + 1) as F / 4.),
        [0., 1., 0.],
    );
    use super::ply::Ply;
    let vc = vc
        .into_iter()
        .map(|rgb| rgb.map(|v| (v * 255.) as u8))
        .collect::<Vec<_>>();

    let fs = fs
        .into_iter()
        .map(|tri| super::FaceKind::Quad(tri))
        .collect::<Vec<_>>();

    let ply = Ply::new(vs, vc, fs);
    use std::fs::File;
    let f = File::create("raw_edge_vis.ply").unwrap();
    ply.write(f).unwrap();
}
