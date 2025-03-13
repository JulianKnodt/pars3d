use super::anim::Animation;
use super::{add, kmul, sub, FaceKind, F, U};

use std::array::from_fn;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Range;

/// Max number of supported UV channels
pub const MAX_UV: usize = 4;

/// Global settings that define the orientation of each scene.
/// The settings depend on the input format.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Axis of the up direction
    pub up_axis: Axis,
    /// Axis of the fwd direction
    pub fwd_axis: Axis,
    /// Axis of the tangent direction, is one of the +/- cross product of up and fwd
    pub tan_axis: Axis,
    /// Scale of the model
    pub scale: F,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            up_axis: Axis::PosY,
            fwd_axis: Axis::NegZ,
            tan_axis: Axis::PosX,
            scale: 1.,
        }
    }
}

impl Settings {
    pub fn coord_system(&self) -> CoordSystem {
        let axes = [self.fwd_axis, self.up_axis, self.tan_axis];
        CoordSystem { axes }
    }
}

/// The usage of a specific texture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    Diffuse,
    Normal,
    Emissive,
    Specular,
    Metallic,
    Roughness,
}

/// Texture for use in a rendering material.
#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    /// Original usage of this texture.
    pub kind: TextureKind,
    /// Uniform multiplier for all texels.
    pub mul: [F; 4],
    /// Image to be sampled when sampling texels.
    pub image: Option<image::DynamicImage>,
    /// Path to original image, if any.
    pub original_path: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Material {
    pub textures: Vec<Texture>,
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Skin {
    pub inv_bind_matrices: Vec<[[F; 4]; 4]>,
    // Node indices
    pub joints: Vec<usize>,
    // Index of skeleton of whole mesh
    pub skeleton: Option<usize>,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecomposedTransform {
    pub scale: [F; 3],
    pub rotation: [F; 3],
    pub translation: [F; 3],
}

impl Default for DecomposedTransform {
    fn default() -> Self {
        Self {
            scale: [1.; 3],
            rotation: [0.; 3],
            translation: [0.; 3],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    Decomposed(DecomposedTransform),
    Matrix([[F; 4]; 4]),
}

impl Default for Transform {
    fn default() -> Self {
        Self::ident_mat()
    }
}

impl Transform {
    pub fn ident_mat() -> Self {
        Transform::Matrix(super::identity::<4>())
    }
    /// Checks if this transform is the identity transformation
    pub fn is_identity(&self) -> bool {
        match self {
            Transform::Matrix(m) => *m == super::identity::<4>(),
            Transform::Decomposed(d) => *d == DecomposedTransform::default(),
        }
    }
    pub fn to_mat(self) -> [[F; 4]; 4] {
        match self {
            Transform::Matrix(m) => m,
            _ => todo!(),
        }
    }
    pub fn to_decomposed(self) -> DecomposedTransform {
        match self {
            Transform::Decomposed(d) => d,
            _ => todo!(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Node {
    pub mesh: Option<usize>,
    pub children: Vec<usize>,

    pub transform: Transform,

    pub skin: Option<usize>,
    pub name: String,

    pub hidden: bool,
}

impl Node {
    pub fn traverse_with_parent<T>(
        &self,
        scene: &Scene,
        parent_val: T,
        visit: &mut impl FnMut(&Node, T) -> T,
    ) where
        T: Copy,
    {
        let new_val = visit(self, parent_val);
        for &c in &self.children {
            scene.nodes[c].traverse_with_parent(scene, new_val, visit);
        }
    }

    pub fn traverse_mut_with_parent<T>(
        &mut self,
        scene: &mut Scene,
        parent_val: T,
        visit: &mut impl FnMut(&mut Node, T) -> T,
    ) where
        T: Copy,
    {
        let new_val = visit(self, parent_val);
        for c in 0..self.children.len() {
            let ci = self.children[c];
            let mut curr_node = std::mem::take(&mut scene.nodes[ci]);
            curr_node.traverse_mut_with_parent(scene, new_val, visit);
            scene.nodes[ci] = curr_node;
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Scene {
    /// Root nodes
    pub root_nodes: Vec<usize>,
    /// All nodes
    pub nodes: Vec<Node>,
    /// Meshes in this scene, order is not relevant for rendering
    pub meshes: Vec<Mesh>,
    /// Materials
    pub materials: Vec<Material>,
    /// Bone structures
    pub skins: Vec<Skin>,
    /// Set of animations
    pub animations: Vec<Animation>,

    /// For an OBJ input, where are the MTL files
    pub(crate) mtllibs: Vec<String>,

    /// Path to the input file
    /// Needed for saving output later
    pub(crate) input_file: String,

    /// Global settings for axis and scale.
    pub settings: Settings,
}

impl Scene {
    /// Traverse the node graph of a scene, with some state from the parent node.
    pub fn traverse_with_parent<T>(
        &self,
        root_init: impl Fn() -> T,
        visit: &mut impl FnMut(&Node, T) -> T,
    ) where
        T: Copy,
    {
        for &ri in &self.root_nodes {
            self.nodes[ri].traverse_with_parent(self, root_init(), visit);
        }
    }

    /// Mutable traversal of the node graph of a scene, with some state from the parent node.
    pub fn traverse_mut_with_parent<T>(
        &mut self,
        root_init: impl Fn() -> T,
        visit: &mut impl FnMut(&mut Node, T) -> T,
    ) where
        T: Copy,
    {
        for i in 0..self.root_nodes.len() {
            let ri = self.root_nodes[i];
            let mut node = std::mem::take(&mut self.nodes[ri]);
            node.traverse_mut_with_parent(self, root_init(), visit);
            self.nodes[ri] = node;
        }
    }

    /// Converts the scale in global settings to some target value, while rescaling the vertices
    /// in each corresponding mesh. Can be used to normalize different input formats which have
    /// different scales.
    pub fn convert_scale_to(&mut self, tgt: F) {
        let mul = self.settings.scale / tgt;
        for m in self.meshes.iter_mut() {
            for v in m.v.iter_mut() {
                *v = kmul(mul, *v);
            }
        }
        self.settings.scale = tgt;
    }

    pub fn num_vertices(&self) -> usize {
        self.meshes.iter().map(|m| m.v.len()).sum::<usize>()
    }
    pub fn num_faces(&self) -> usize {
        self.meshes.iter().map(|m| m.f.len()).sum::<usize>()
    }

    pub fn set_axis_to(&mut self, tgt_axes: &CoordSystem) {
        let curr_axes = self.settings.coord_system();
        for m in self.meshes.iter_mut() {
            for v in m.v.iter_mut() {
                *v = curr_axes.convert_to(tgt_axes, *v);
            }
            // TODO also handle vertex normals here.
        }
    }

    /// Converts this scene into a flattened mesh which can then be repopulated back into a
    /// scene later.
    pub fn into_flattened_mesh(&self) -> Mesh {
        let has_joints = self.meshes.iter().any(|m| {
            !m.joint_weights.is_empty() && m.joint_weights.iter().any(|&jw| jw != [0.; 4])
        });
        let mut out = Mesh::default();
        for (mi, m) in self.meshes.iter().enumerate() {
            let curr_vertex_offset = out.v.len();
            out.v.extend(m.v.iter().copied());
            for chan in 0..MAX_UV {
                out.uv[chan].extend(m.uv[chan].iter().copied());
            }
            out.n.extend(m.n.iter().copied());
            out.vert_colors.extend(m.vert_colors.iter().copied());
            let curr_f = out.f.len();
            out.f.extend(m.f.iter().map(|f| {
                let mut f = f.clone();
                f.map(|vi| vi + curr_vertex_offset);
                f
            }));
            out.face_mesh_idx.extend(m.f.iter().map(|_| mi));
            out.face_mat_idx.extend(
                m.face_mat_idx
                    .iter()
                    .map(|(f, m)| ((f.start + curr_f)..(f.end + curr_f), *m)),
            );

            // always pad to equivalent number of joints and vertices OR 0 joints.
            if has_joints {
                if m.joint_idxs.is_empty() {
                    out.joint_idxs.extend((0..m.v.len()).map(|_| [0; 4]));
                    out.joint_weights.extend((0..m.v.len()).map(|_| [0.; 4]));
                } else {
                    assert_eq!(m.v.len(), m.joint_idxs.len());
                    out.joint_idxs.extend(m.joint_idxs.iter().copied());
                    assert_eq!(m.v.len(), m.joint_weights.len());
                    out.joint_weights.extend(m.joint_weights.iter().copied());
                }
            }
        }
        if !out.joint_idxs.is_empty() {
            assert_eq!(out.joint_idxs.len(), out.joint_weights.len());
            assert_eq!(out.joint_idxs.len(), out.v.len());
        }
        out
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Vertex positions.
    pub v: Vec<[F; 3]>,
    /// Set of UVs.
    pub uv: [Vec<[F; 2]>; MAX_UV],

    /// Normals.
    pub n: Vec<[F; 3]>,

    /// Vertex colors.
    pub vert_colors: Vec<[F; 3]>,

    /// Faces.
    pub f: Vec<FaceKind>,
    /// Which mesh did this face come from?
    /// Used when flattening a scene into a single mesh.
    pub face_mesh_idx: Vec<usize>,

    /// Map of ranges for each face that correspond to a specific material
    pub face_mat_idx: Vec<(Range<usize>, usize)>,

    /// For each vertex, the index of bone influences.
    /// 1-1 relation between vertices and joint/idxs weights.
    pub joint_idxs: Vec<[u16; 4]>,
    /// The weight of each bone's influence. 0 indicates no influence.
    pub joint_weights: Vec<[F; 4]>,

    /// Name of this mesh.
    pub name: String,
}

impl Mesh {
    /// Flips the 2nd channel of each UV.
    pub fn flip_uv_v(&mut self) {
        for uv_chan in self.uv.iter_mut() {
            for uvs in uv_chan.iter_mut() {
                uvs[1] = 1. - uvs[1];
            }
        }
    }

    /// Converts a single mesh into a scene
    pub fn into_scene(mut self) -> Scene {
        self.face_mesh_idx.clear();
        Scene {
            meshes: vec![self],
            ..Default::default()
        }
    }
    /// Computes the number of chords in a mesh,
    /// and accepts a function which is run on each chord's length.
    pub fn num_quad_chords(&self, mut chord_size_fn: impl FnMut(usize)) -> usize {
        use std::collections::HashSet;
        let mut seen_edges: HashSet<[usize; 2]> = HashSet::new();
        let mut edge_face_adj: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
        for ([e0, e1], f) in self.edges() {
            let e = std::cmp::minmax(e0, e1);
            edge_face_adj.entry(e).or_default().push(f);
        }
        let mut num_chords = 0;
        for &[e0, e1] in edge_face_adj.keys() {
            let e = std::cmp::minmax(e0, e1);
            if !seen_edges.insert(e) {
                continue;
            }
            num_chords += 1;
            let mut curr_chord_len = 0;
            let mut buf: Vec<[usize; 2]> = vec![e];
            while let Some([n0, n1]) = buf.pop() {
                let n = std::cmp::minmax(n0, n1);
                curr_chord_len += 1;
                for &adj in &edge_face_adj[&n] {
                    let face = &self.f[adj];
                    let Some([oe_0, oe_1]) = face.quad_opp_edge(n0, n1) else {
                        continue;
                    };
                    let opp_e = std::cmp::minmax(oe_0, oe_1);
                    if !seen_edges.insert(opp_e) {
                        continue;
                    }
                    buf.push(opp_e);
                }
            }
            chord_size_fn(curr_chord_len);
        }
        num_chords
    }

    /// Returns an iterator over (edge, face) pairs in this mesh.
    /// Each edge will be visited multiple times, equaling the number of its adjacent faces.
    pub fn edges(&self) -> impl Iterator<Item = ([usize; 2], usize)> + '_ {
        self.f.iter().enumerate().flat_map(|(fi, f)| {
            let f_sl = f.as_slice();
            f_sl.array_windows::<2>()
                .copied()
                .chain(std::iter::once([*f_sl.last().unwrap(), f_sl[0]]))
                .map(move |e| (e, fi))
        })
    }
    /// Returns the material for a given face if any.
    pub fn mat_for_face(&self, fi: usize) -> Option<usize> {
        self.face_mat_idx
            .iter()
            .find_map(|(fr, mi)| fr.contains(&fi).then_some(*mi))
    }
    /// The number of triangles in this mesh, after triangulation.
    pub fn num_tris(&self) -> usize {
        self.f.iter().map(|f| f.num_tris()).sum::<usize>()
    }

    /// Triangulates this mesh in an arbitrary order.
    /// Will allocate if not all faces are triangles.
    /// Order of faces may not be preserved.
    pub fn triangulate(&mut self) {
        let mut i = 0;
        // nifty little method which doesn't require an extra buffer
        while i < self.f.len() {
            if self.f[i].len() <= 3 {
                i += 1;
                continue;
            }
            let f = self.f.swap_remove(i);
            self.f.extend(f.as_triangle_fan().map(FaceKind::Tri));
        }
    }

    /// Normalize this mesh's geometry to lay within [-1, 1].
    /// Outputs scale and translation to reposition back to the original dimension.
    pub fn normalize(&mut self) -> (F, [F; 3]) {
        // Normalize the geometry of this mesh to lay in the unit box.
        let [l, h] = self
            .v
            .iter()
            .fold([[F::INFINITY; 3], [F::NEG_INFINITY; 3]], |[l, h], n| {
                [from_fn(|i| l[i].min(n[i])), from_fn(|i| h[i].max(n[i]))]
            });
        let center = kmul(0.5, add(l, h));
        for v in &mut self.v {
            *v = sub(*v, center);
        }
        let largest_val = self
            .v
            .iter()
            .fold(0., |m, v| v.iter().fold(m, |m, c| c.abs().max(m)));
        let scale = if largest_val == 0. {
            1.
        } else {
            largest_val.recip()
        };
        for v in &mut self.v {
            *v = kmul(scale, *v);
        }
        (scale, center)
    }
    /// Given a scale and translation output from normalization, reset the geometry to its
    /// original position.
    pub fn denormalize(&mut self, scale: F, trans: [F; 3]) {
        assert_ne!(scale, 0.);
        let inv_scale = scale.recip();
        for v in &mut self.v {
            *v = add(kmul(inv_scale, *v), trans);
        }
    }
    /// After flattening a scene into a mesh, repopulate the original scene with a modified
    /// flattened mesh.
    pub fn repopulate_scene(&self, scene: &mut Scene) {
        assert_eq!(
            self.face_mesh_idx.len(),
            self.f.len(),
            "Must specify which mesh belongs to which face"
        );

        // TODO maybe should also check that the number of meshes in the scene is greater than the
        // max number of meshes in the original mesh

        // mesh -> original_vertex_idx -> new_vertex_idx
        let mut vertex_map = vec![HashMap::new(); scene.meshes.len()];
        scene.meshes.fill_with(Default::default);
        // material for each mesh for each face
        let mut mat_map = vec![vec![]; scene.meshes.len()];

        for (fi, f) in self.f.iter().enumerate() {
            let mi = self.face_mesh_idx[fi];
            let mesh = &mut scene.meshes[mi];
            mat_map[mi].push(self.mat_for_face(fi));

            let mut f = f.clone();
            f.map(|flat_vi| {
                let new_vert_ins = || {
                    let vi = mesh.v.len();
                    mesh.v.push(self.v[flat_vi]);
                    for chan in 0..MAX_UV {
                        if let Some(&uv) = self.uv[chan].get(flat_vi) {
                            mesh.uv[chan].push(uv);
                            assert_eq!(mesh.v.len(), mesh.uv[chan].len());
                        }
                    }

                    if let Some(&n) = self.n.get(flat_vi) {
                        mesh.n.push(n);
                        assert_eq!(mesh.v.len(), mesh.n.len());
                    }

                    if let Some(&ji) = self.joint_idxs.get(flat_vi) {
                        mesh.joint_idxs.push(ji);
                        assert_eq!(mesh.v.len(), mesh.joint_idxs.len());
                    }

                    if let Some(&jw) = self.joint_weights.get(flat_vi) {
                        mesh.joint_weights.push(jw);
                        assert_eq!(mesh.v.len(), mesh.joint_weights.len());
                    }

                    vi
                };
                *vertex_map[mi].entry(flat_vi).or_insert_with(new_vert_ins)
            });

            mesh.f.push(f);
        }

        for (mi, mesh) in scene.meshes.iter_mut().enumerate() {
            mesh.face_mat_idx = convert_opt_usize(&mat_map[mi]);
        }
    }

    /// Strips all non-geometry attributes from this mesh, leaving only the geometry.
    pub fn geometry_only(&mut self) {
        let v = std::mem::take(&mut self.v);

        let mut new_v = vec![];

        // original vertex position to new vertex index
        let mut new_map: HashMap<[U; 3], usize> = HashMap::new();

        fn key(v: [F; 3]) -> [U; 3] {
            v.map(F::to_bits)
        }

        for f in self.f.iter_mut() {
            for vi in f.as_mut_slice() {
                let k = key(v[*vi]);
                *vi = match new_map.entry(k) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(vac) => {
                        let new_idx = new_v.len();
                        new_v.push(v[*vi]);
                        vac.insert(new_idx);
                        new_idx
                    }
                }
            }
        }

        *self = Mesh {
            v: new_v,
            f: std::mem::take(&mut self.f),
            ..Default::default()
        }
    }
    pub fn normalize_joint_weights(&mut self) {
        for ws in &mut self.joint_weights {
            let sum = ws.iter().sum::<F>();
            if sum <= 0. {
                continue;
            }
            *ws = ws.map(|v| v / sum);
        }
    }
}

// For converting optional material per face index to a range of faces.
pub fn convert_opt_usize(s: &[Option<usize>]) -> Vec<(Range<usize>, usize)> {
    let mut out = vec![];
    for (i, mati) in s.iter().enumerate() {
        let &Some(mati) = mati else {
            continue;
        };
        match out.last_mut() {
            None => out.push((i..(i + 1), mati)),
            Some((v, p_mati)) => {
                if v.end == i && mati == *p_mati {
                    v.end += 1;
                } else {
                    out.push((i..(i + 1), mati));
                }
            }
        }
    }

    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    /// +X
    PosX,
    /// +Y
    PosY,
    /// +Z
    PosZ,
    /// -X
    NegX,
    /// -Y
    NegY,
    /// -Z
    NegZ,
}

impl Axis {
    pub fn is_parallel_to(&self, o: &Self) -> bool {
        match (self, o) {
            (Self::PosX | Self::NegX, Self::PosX | Self::NegX) => true,
            (Self::PosY | Self::NegY, Self::PosY | Self::NegY) => true,
            (Self::PosZ | Self::NegZ, Self::PosZ | Self::NegZ) => true,
            _ => false,
        }
    }
    pub fn cross(&self, o: &Self) -> Option<Axis> {
        let v = match (self, o) {
            (Self::PosX, Self::PosY) | (Self::NegX, Self::NegY) => Self::PosZ,
            (Self::PosX, Self::NegY) | (Self::NegX, Self::PosY) => Self::NegZ,

            (Self::PosY, Self::PosX) | (Self::NegY, Self::NegX) => Self::NegZ,
            (Self::PosY, Self::NegX) | (Self::NegY, Self::PosX) => Self::PosZ,

            //
            (Self::PosX, Self::PosZ) | (Self::NegX, Self::NegZ) => Self::NegY,
            (Self::PosX, Self::NegZ) | (Self::NegX, Self::PosZ) => Self::PosY,

            (Self::PosZ, Self::PosX) | (Self::NegZ, Self::NegX) => Self::PosY,
            (Self::PosZ, Self::NegX) | (Self::NegZ, Self::PosX) => Self::NegY,

            //
            (Self::PosY, Self::PosZ) | (Self::NegY, Self::NegZ) => Self::NegX,
            (Self::PosY, Self::NegZ) | (Self::NegY, Self::PosZ) => Self::PosX,

            (Self::PosZ, Self::PosY) | (Self::NegZ, Self::NegY) => Self::PosX,
            (Self::PosZ, Self::NegY) | (Self::NegZ, Self::PosY) => Self::NegX,

            _ => return None,
        };
        Some(v)
    }
    pub fn flip(&self) -> Axis {
        match self {
            Self::PosX => Self::NegX,
            Self::NegX => Self::PosX,

            Self::PosY => Self::NegY,
            Self::NegY => Self::PosY,

            Self::PosZ => Self::NegZ,
            Self::NegZ => Self::PosZ,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordSystem {
    axes: [Axis; 3],
}

impl CoordSystem {
    pub fn new(fwd: Axis, up: Axis, rh: bool) -> Self {
        assert!(
            !fwd.is_parallel_to(&up),
            "{fwd:?} {up:?} should not lie on the same axis."
        );
        let right = fwd.cross(&up).unwrap();
        let tan = if rh { right } else { right.flip() };

        Self {
            axes: [fwd, up, tan],
        }
    }
    pub fn standard() -> Self {
        let fwd = Axis::PosX;
        let up = Axis::PosY;
        let tan = Axis::PosZ;
        Self {
            axes: [fwd, up, tan],
        }
    }

    /// Converts a value in one coordinate system into another coordinate system.
    pub fn convert_to(&self, tgt: &Self, v: [F; 3]) -> [F; 3] {
        let mut out = [0.; 3];
        for i in 0..3 {
            let sa = self.axes[i];
            for j in 0..3 {
                let ta = tgt.axes[j];
                if !sa.is_parallel_to(&ta) {
                    continue;
                }
                if sa == ta.flip() {
                    out[j] = -v[i];
                } else {
                    out[j] = v[i];
                }
            }
        }
        out
    }
}
