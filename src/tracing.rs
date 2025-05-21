use super::face::Barycentric;
use super::quat::{quat_from_to, quat_rot};
use super::visualization::colored_wireframe;
use super::{
    add, barycentric_3d, dir_to_barycentric, dist, edges as edges_iter, kmul, normalize, sub,
    FaceKind, F,
};

/*
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WidthKind {
  Constant(F),
  LinearDecay {
    initial: F,
    decay_rate: F,
  },
}
*/

/// Describes a curve on the surface of a triangle mesh
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    // The barycentric coordinate within the first face
    start: Barycentric,
    // index of first face of the curve (not first tri, face)
    start_face: usize,
    /// Direction in barycentric coordinates (v0, v1) on the surface of a mesh.
    direction: [F; 2],

    /// Width of the edge
    width: F,

    /// Length in world-space distance
    length: F,

    /// How much to rotate the direction by length (coarsely approximated)
    bend_amt: F,
}

/// Trace a curve along the surface of this mesh.
pub fn trace_curve<'a>(
    vs: &[[F; 3]],
    fs: &[FaceKind],
    // For each edge, returns adjacent faces
    edge_adj: impl Fn([usize; 2]) -> &'a [usize],

    curve: Curve,
) -> (Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut rem_len = curve.length;

    let mut curr_bary = curve.start;
    let mut curr_face = curve.start_face;

    let mut curr_dir = normalize(curve.direction);

    let pos = fs[curr_face]
        .map_kind(|vi| vs[vi])
        .from_barycentric(curr_bary);

    let mut pos = vec![pos];
    let mut edges = vec![];
    loop {
        // need to find intersection with other triangle edge
        let (tri_idx, coords) = curr_bary.tri_idx_and_coords();
        let tri = fs[curr_face].as_triangle_fan().nth(tri_idx).unwrap();
        // find intersection with x = 0, y = 0, (x+y) = 1
        let isect_x = -coords[0] / curr_dir[0];
        let isect_y = -coords[1] / curr_dir[1];
        let isect_xy = (1. - coords[0] - coords[1]) / (curr_dir[0] + curr_dir[1]);
        let uv = [coords[0], coords[1]];
        assert!([isect_x, isect_y, isect_xy].iter().copied().any(F::is_finite));
        let nearest = [isect_x, isect_y, isect_xy]
            .into_iter()
            .filter(|v| v.is_finite())
            // only allow forward direction (w/ some eps so can't intersect repeat edge)
            .filter(|&v| v > 1e-5)
            .min_by(|a, b| a.partial_cmp(&b).unwrap());
        let Some(nearest) = nearest else {
            curr_dir = curr_dir.map(core::ops::Neg::neg);
            continue;
        };
        let new_pos = add(uv, kmul(nearest, curr_dir));
        let edge_bary = [
            new_pos[0],
            new_pos[1],
            (1. - new_pos[0] - new_pos[1]),
        ];

        let new_global_pos = FaceKind::Tri(tri)
            .map_kind(|vi| vs[vi])
            .from_barycentric_tri(edge_bary.map(|v| v.clamp(0., 1.)));

        // TODO here need to cut if off earlier if the distance exceeds the length of the curve.
        let seg_len = dist(new_global_pos, *pos.last().unwrap());
        let new_len = rem_len - seg_len;
        if new_len <= 0. {
            let t = rem_len / seg_len;
            assert!((0.0..=1.).contains(&t));
            let prev_pos = *pos.last().unwrap();
            let dir = kmul(t, sub(new_global_pos, prev_pos));
            pos.push(add(prev_pos, dir));
            edges.push([pos.len() - 2, pos.len() - 1]);
            break;
        }
        rem_len = new_len;

        pos.push(new_global_pos);
        edges.push([pos.len() - 2, pos.len() - 1]);

        let min_bary = edge_bary
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        let opp_e = match min_bary {
            0 => [1, 2],
            1 => [0, 2],
            2 => [0, 1],
            _ => unreachable!(),
        };
        let opp_e = opp_e.map(|o| tri[o]);

        // TODO test if it's an internal edge of a face, then need to have some different
        // operation?

        let opp_f = edge_adj(opp_e)
            .into_iter()
            .copied()
            .find(|&v| v != curr_face);
        // for now break until hit a boundary
        let Some(opp_f) = opp_f else {
            break;
        };
        curr_face = opp_f;
        // find which triangle of this face contains the edge traversed
        let (ti, new_tri) = fs[opp_f]
            .as_triangle_fan()
            .enumerate()
            .find(|(_ti, tri)| {
                edges_iter(tri).any(|[e0, e1]| [e0, e1] == opp_e || [e1, e0] == opp_e)
            })
            .unwrap();
        let new_bary = barycentric_3d(new_global_pos, new_tri.map(|vi| vs[vi]));
        curr_bary = Barycentric::new(&fs[opp_f], ti, new_bary);

        let np = pos.len();
        let new_world_dir = normalize(sub(pos[np - 1], pos[np - 2]));
        let curr_n = FaceKind::Tri(tri).normal(&vs);
        let new_n = FaceKind::Tri(new_tri).normal(&vs);
        let new_world_dir = normalize(quat_rot(new_world_dir, quat_from_to(curr_n, new_n)));
        curr_dir = dir_to_barycentric(new_world_dir, new_tri.map(|vi| vs[vi]));
        if curve.bend_amt != 0. {
          let rot = super::rot_matrix_2d(seg_len * curve.bend_amt);
          curr_dir = super::matmul_2d(rot, curr_dir);
        }
    }

    let (v, _, fs) = colored_wireframe(edges.into_iter(), |vi| pos[vi], |_| [0.; 3], curve.width);
    (v, fs)
}

#[test]
fn test_trace() {
    let vs = [
        [-0.5, 0., 0.],
        [0.4, 0., 0.],
        [-0.1, 1., 0.],
        [0.5, 0.7, -0.3],
    ];
    let fs = [FaceKind::Tri([0, 1, 2]), FaceKind::Tri([1, 3, 2])];

    let bary_dir = super::dir_to_barycentric([0.5, 0.0, 0.0], std::array::from_fn(|i| vs[i]));
    let curve = Curve {
        start: Barycentric::Tri([0.33333, 0.33333, 1. - (0.33333 + 0.33333)]),
        start_face: 0,

        direction: bary_dir,

        length: 100.,
        width: 3e-3,

        bend_amt: 0.,
    };

    let (curve_v, curve_f) = trace_curve(
        &vs,
        &fs,
        |[e0, e1]| {
            if std::cmp::minmax(e0, e1) == [1, 2] {
                &[1, 2]
            } else {
                &[]
            }
        },
        curve,
    );

    let mut m = super::Mesh {
        v: vs.to_vec(),
        vert_colors: vec![[1.; 3]; vs.len()],
        f: fs.to_vec(),
        ..Default::default()
    };

    let black_verts = vec![[0.; 3]; curve_v.len()];
    let mut wf = super::visualization::wireframe_to_mesh((curve_v, black_verts, curve_f));
    m.append(&mut wf);

    super::save("curve_test.ply", &m.into_scene()).expect("Failed to save");
}

#[test]
fn test_trace_sphere() {
    let mut m = super::load("icosphere.obj").unwrap().into_flattened_mesh();
    m.geometry_only();
    m.vert_colors = vec![[1.; 3]; m.v.len()];
    let edge_adj = m.edge_adj_map();

    let curve = Curve {
        start: Barycentric::Tri([0., 0., 1.]),
        start_face: 0,
        direction: normalize([0.5, 0.1]),
        length: 5000.,
        width: 3e-3,
        bend_amt: 0.,
    };

    let (curve_v, curve_f) = trace_curve(
        &m.v,
        &m.f,
        |[e0, e1]| &edge_adj[&std::cmp::minmax(e0, e1)],
        curve,
    );
    let black_verts = vec![[0.; 3]; curve_v.len()];
    let mut wf = super::visualization::wireframe_to_mesh((curve_v, black_verts, curve_f));
    m.append(&mut wf);
    m.uv[0].clear();
    m.n.clear();

    super::save("sphere_curve_test.ply", &m.into_scene()).expect("Failed to save");
}
