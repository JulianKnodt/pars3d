use super::face::Barycentric;
use super::quat::{quat_from_to, quat_rot};
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

/// How to define color on a curve
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorKind {
    /// Constant color on the whole curve
    Constant([F; 3]),

    /// Linearly interpolate the color along the curve
    Linear([F; 3], [F; 3]),
    // TODO add more complex functions here?
}

impl ColorKind {
    pub fn starting_color(&self) -> [F; 3] {
        match self {
            &ColorKind::Constant(c) => c,
            &ColorKind::Linear(s, _) => s,
        }
    }
    pub fn ending_color(&self) -> [F; 3] {
        match self {
            &ColorKind::Constant(c) => c,
            &ColorKind::Linear(_, e) => e,
        }
    }
    pub fn lerp(&self, t: F) -> [F; 3] {
        match self {
            &ColorKind::Constant(c) => c,
            &ColorKind::Linear(s, e) => add(kmul(1. - t, s), kmul(t, e)),
        }
    }
    fn split(&self) -> (ColorKind, ColorKind) {
        match self {
            &ColorKind::Constant(c) => (ColorKind::Constant(c), ColorKind::Constant(c)),
            &ColorKind::Linear(s, e) => {
                let mid = self.lerp(0.5);
                (ColorKind::Linear(s, mid), ColorKind::Linear(mid, e))
            }
        }
    }
}

impl Default for ColorKind {
    fn default() -> Self {
        ColorKind::Constant([0.; 3])
    }
}

/// Describes a curve on the surface of a triangle mesh
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    // The barycentric coordinate within the first face
    pub start: Barycentric,
    // index of first face of the curve (not first tri, face)
    pub start_face: usize,
    /// Direction in barycentric coordinates (v0, v1) on the surface of a mesh.
    pub direction: [F; 2],

    /// Width of the edge
    pub width: F,

    /// Length in world-space distance
    pub length: F,

    /// How much to rotate the direction by length (coarsely approximated)
    /// Expressed in radians, but note that it will be scaled by distance, so test large values
    /// for style.
    pub bend_amt: F,

    pub color: ColorKind,
}

pub struct CurveSet {
    pub count: usize,
}

/*
impl Curve {
    pub fn new_on_edge(v: &[[F;3]], f: &FaceKind, mut e: [usize; 2], dir: [F; 3]) -> Self {
        let (ei, _) = f.edges().enumerate().find(|(_, fe)| {
            let flip = [fe[1], fe[0]];
            if e == flip {
                e = e.swap(0, 1);
                return fe;
            }
            e == fe
        });
        let bary_dir = super::dir_to_barycentric(dir, std::array::from_fn(|i| vs[i]));
    }
}
*/

pub fn trace_curve_from_mid<'a>(
    vs: &[[F; 3]],
    fs: &[FaceKind],
    // For each edge, returns adjacent faces
    edge_adj: impl Fn([usize; 2]) -> &'a [usize],

    mut curve: Curve,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    for c in curve.start.coords_mut() {
        *c = c.clamp(1e-2, 0.99);
    }
    curve.start.normalize();

    curve.length /= 2.;
    let (c0, c1) = curve.color.split();
    //curve.color = c0;
    curve.color = ColorKind::Constant([0.; 3]);
    let (mut v0, mut vc0, mut f0) = trace_curve(vs, fs, &edge_adj, curve);
    //curve.color = c1;
    curve.color = ColorKind::Constant([1.; 3]);
    curve.direction = curve.direction.map(|d| -d);

    let (mut v1, mut vc1, mut f1) = trace_curve(vs, fs, &edge_adj, curve);
    let o = vc0.len();
    for vis in f1.iter_mut() {
        *vis = vis.map(|vi| vi + o)
    }
    v0.append(&mut v1);
    vc0.append(&mut vc1);
    f0.append(&mut f1);
    (v0, vc0, f0)
}

/// Trace a curve along the surface of this mesh.
pub fn trace_curve<'a>(
    vs: &[[F; 3]],
    fs: &[FaceKind],
    // For each edge, returns adjacent faces
    edge_adj: impl Fn([usize; 2]) -> &'a [usize],

    curve: Curve,
) -> (Vec<[F; 3]>, Vec<[F; 3]>, Vec<[usize; 4]>) {
    let mut rem_len = curve.length;

    let mut curr_bary = curve.start;
    let mut curr_face = curve.start_face;

    let mut curr_dir = normalize(curve.direction);

    let pos = fs[curr_face]
        .map_kind(|vi| vs[vi])
        .from_barycentric(curr_bary);

    let mut pos = vec![pos];
    let mut colors = vec![curve.color.starting_color()];

    let mut did_flip = false;
    loop {
        // need to find intersection with other triangle edge
        let (tri_idx, coords) = curr_bary.tri_idx_and_coords();
        let tri = fs[curr_face].as_triangle_fan().nth(tri_idx).unwrap();
        // find intersection with x = 0, y = 0, (x+y) = 1
        let isect_x = -coords[0] / curr_dir[0];
        let isect_y = -coords[1] / curr_dir[1];
        let isect_xy = (1. - coords[0] - coords[1]) / (curr_dir[0] + curr_dir[1]);
        let uv = [coords[0], coords[1]];
        let isects = [isect_x, isect_y, isect_xy];
        assert!(
            isects.iter().copied().any(F::is_finite),
            "{isects:?} {uv:?} {curr_dir:?}"
        );
        let nearest = isects
            .into_iter()
            .filter(|v| v.is_finite())
            // only allow forward direction (w/ some eps so can't intersect repeat edge)
            .filter(|&v| v > 1e-5)
            .min_by(|a, b| a.partial_cmp(&b).unwrap());
        let Some(nearest) = nearest else {
            if did_flip {
                return (vec![], vec![], vec![]);
            }
            did_flip = true;
            curr_dir = curr_dir.map(core::ops::Neg::neg);
            continue;
        };
        did_flip = false;
        let new_pos = add(uv, kmul(nearest, curr_dir));
        let edge_bary = [new_pos[0], new_pos[1], (1. - new_pos[0] - new_pos[1])];

        let new_global_pos = FaceKind::Tri(tri)
            .map_kind(|vi| vs[vi])
            .from_barycentric_tri(edge_bary.map(|v| v.clamp(0., 1.)));

        // TODO here need to cut if off earlier if the distance exceeds the length of the curve.
        let seg_len = dist(new_global_pos, *pos.last().unwrap());
        if seg_len <= 1e-8 {
            return (vec![], vec![], vec![]);
        }
        let new_len = rem_len - seg_len;
        if new_len <= 0. {
            let t = rem_len / seg_len;
            assert!((0.0..=1.).contains(&t));
            let prev_pos = *pos.last().unwrap();
            let dir = kmul(t, sub(new_global_pos, prev_pos));
            pos.push(add(prev_pos, dir));
            colors.push(curve.color.ending_color());
            break;
        }
        rem_len = new_len;

        pos.push(new_global_pos);
        let t = 1. - rem_len / curve.length;
        colors.push(curve.color.lerp(t));

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
            //break;
            return (vec![], vec![], vec![]);
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
        if new_world_dir == [0.; 3] {
            // TODO make this check other faces until a non-degenerate face is found
            return (vec![], vec![], vec![]);
        }
        assert_ne!(new_world_dir, [0., 0., 0.]);
        let curr_n = FaceKind::Tri(tri).normal(&vs);
        let new_n = FaceKind::Tri(new_tri).normal(&vs);
        let new_world_dir = normalize(quat_rot(new_world_dir, quat_from_to(curr_n, new_n)));
        curr_dir = dir_to_barycentric(new_world_dir, new_tri.map(|vi| vs[vi]));
        if curve.bend_amt != 0. {
            let rot = super::rot_matrix_2d(seg_len * curve.bend_amt);
            curr_dir = super::matmul_2d(rot, curr_dir);
        }
    }

    let nv = pos.len();
    if dist(pos[0], pos[nv - 1]) < 0.8 * curve.length
        || dist(pos[0], pos[nv - 2]) < 0.8 * curve.length
    {
        return (vec![], vec![], vec![]);
    }

    super::visualization::per_vertex_colored_wireframe(
        pos.len(),
        |vi| (pos[vi], colors[vi]),
        curve.width,
    )
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
        color: Default::default(),
    };

    let (curve_v, _, curve_f) = trace_curve(
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
    let edge_adj = m.edge_kinds();

    let curve = Curve {
        start: Barycentric::Tri([0., 0., 1.]),
        start_face: 0,
        direction: normalize([0.5, 0.1]),
        length: 5000.,
        width: 3e-3,
        bend_amt: 0.,
        color: Default::default(),
    };

    let (curve_v, _, curve_f) = trace_curve(
        &m.v,
        &m.f,
        |[e0, e1]| edge_adj[&std::cmp::minmax(e0, e1)].as_slice(),
        curve,
    );
    let black_verts = vec![[0.; 3]; curve_v.len()];
    let mut wf = super::visualization::wireframe_to_mesh((curve_v, black_verts, curve_f));
    m.append(&mut wf);
    m.uv[0].clear();
    m.n.clear();

    super::save("sphere_curve_test.ply", &m.into_scene()).expect("Failed to save");
}
