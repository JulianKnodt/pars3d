use super::anim::Animation;
use super::{F, FaceKind, U, add, divk, kmul, sub};

use std::array::from_fn;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
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

impl Texture {
    pub fn new(
        kind: TextureKind,
        mul: [F; 4],
        image: Option<image::DynamicImage>,
        original_path: String,
    ) -> Self {
        Texture {
            kind,
            mul,
            image,
            original_path,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Material {
    pub textures: Vec<usize>,
    pub name: String,
    pub path: String,
}

impl Material {
    /// Gets the texture from this material of a given kind.
    pub fn textures_by_kind<'a: 'c, 'b: 'c, 'c>(
        &'a self,
        textures: &'b [Texture],
        kind: TextureKind,
    ) -> impl Iterator<Item = &'c Texture> + 'c {
        self.textures
            .iter()
            .map(|&ti| &textures[ti])
            .filter(move |t| t.kind == kind)
    }
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
    /// Textures which can be reused by materials,
    pub textures: Vec<Texture>,
    /// Materials
    pub materials: Vec<Material>,
    /// Bone structures
    pub skins: Vec<Skin>,
    /// Set of animations
    pub animations: Vec<Animation>,

    // /// For an OBJ input, where are the MTL files
    //pub(crate) mtllibs: Vec<String>,
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
    /// Number of faces in this scene
    pub fn num_faces(&self) -> usize {
        self.meshes.iter().map(|m| m.f.len()).sum::<usize>()
    }
    /// Number of edges in this scene (allocates).
    pub fn num_edges(&self) -> usize {
        let mut edges: HashSet<[usize; 2]> = HashSet::new();
        for m in self.meshes.iter() {
            for f in &m.f {
                for e in f.edges_ord() {
                    edges.insert(e);
                }
            }
        }
        edges.len()
    }
    /// Number of triangles in this scene
    pub fn num_tris(&self) -> usize {
        self.meshes.iter().map(Mesh::num_tris).sum::<usize>()
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
                f.remap(|vi| vi + curr_vertex_offset);
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

            out.vertex_attrs.extend(&m.vertex_attrs);
        }
        if !out.joint_idxs.is_empty() {
            assert_eq!(out.joint_idxs.len(), out.joint_weights.len());
            assert_eq!(out.joint_idxs.len(), out.v.len());
        }
        out
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct SphHarmonicCoeff {
    order1: [F; 3],
    order2: [F; 5],
    order3: [F; 7],
}

impl SphHarmonicCoeff {
    pub fn get_mut(&mut self, i: usize) -> Option<&mut F> {
        let v = if i < 3 {
            &mut self.order1[i]
        } else if i < 8 {
            &mut self.order2[i - 3]
        } else if i < 15 {
            &mut self.order3[i - 8]
        } else {
            return None;
        };
        Some(v)
    }

    pub fn get(&self, i: usize) -> Option<&F> {
        let v = if i < 3 {
            &self.order1[i]
        } else if i < 8 {
            &self.order2[i - 3]
        } else if i < 15 {
            &self.order3[i - 8]
        } else {
            return None;
        };
        Some(v)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VertexAttrs {
    pub height: Vec<F>,
    pub tangent: Vec<[F; 3]>,
    pub bitangent: Vec<[F; 3]>,

    pub opacity: Vec<F>,
    pub scale: Vec<[F; 3]>,
    pub rot: Vec<[F; 4]>,
    pub sph_harmonic_coeff: Vec<[SphHarmonicCoeff; 3]>,
}

impl VertexAttrs {
    /// Truncates all associated attributes
    pub fn truncate(&mut self, l: usize) {
        self.height.truncate(l);
        self.tangent.truncate(l);
        self.bitangent.truncate(l);
    }

    pub fn extend(&mut self, o: &Self) {
        self.height.extend(o.height.iter().copied());
        self.tangent.extend(o.tangent.iter().copied());
        self.bitangent.extend(o.bitangent.iter().copied());

        self.opacity.extend(o.opacity.iter().copied());
        self.scale.extend(o.scale.iter().copied());
        self.rot.extend(o.rot.iter().copied());
        self.sph_harmonic_coeff
            .extend(o.sph_harmonic_coeff.iter().copied());
    }
}

/// Enum to represent polylines in a mesh. 2 element lines are special cased to avoid heap
/// allocations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Line {
    Standard([usize; 2]),
    Poly(Vec<usize>),
}

impl Line {
    pub fn new(pts: &[usize]) -> Self {
        if let &[a, b] = pts {
            return Self::new_from_endpoints(a, b);
        }
        todo!()
    }
    #[inline]
    pub fn new_from_endpoints(a: usize, b: usize) -> Self {
        Self::Standard(std::cmp::minmax(a, b))
    }
    pub fn as_slice(&self) -> &[usize] {
        match self {
            Self::Standard(l) => l.as_slice(),
            Self::Poly(p) => p.as_slice(),
        }
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

    /// Lines without corresponding faces.
    pub l: Vec<Line>,

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

    /// Additional vertex attributes to store on each vertex of a mesh
    pub vertex_attrs: VertexAttrs,

    /// Name of this mesh.
    pub name: String,
}

impl Mesh {
    pub fn new_geometry(v: Vec<[F; 3]>, f: Vec<FaceKind>) -> Self {
        Self {
            v,
            f,
            ..Default::default()
        }
    }
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
    /// If the mesh uses only a single material, returns that material.
    pub fn single_mat(&self) -> Option<usize> {
        if self.face_mat_idx.is_empty() {
            return None;
        }

        let mat0 = self.face_mat_idx[0].1;
        if self.face_mat_idx.len() == 1 {
            return Some(mat0);
        }

        self.face_mat_idx[1..]
            .iter()
            .all(|v| v.1 == mat0)
            .then_some(mat0)
    }

    /// Assigns a single material to this mesh.
    /// Should correspond to a material in the scene.
    pub fn assign_single_mat(&mut self, mati: usize) {
        self.face_mat_idx = vec![(0..self.f.len(), mati)];
    }
    /// The number of triangles in this mesh, after triangulation.
    pub fn num_tris(&self) -> usize {
        self.f.iter().map(|f| f.num_tris()).sum::<usize>()
    }

    /// Triangulates this mesh in an arbitrary order.
    /// Will allocate if not all faces are triangles.
    /// Order of faces may not be preserved.
    pub fn triangulate(&mut self, base: usize) {
        self.triangulate_with_new_edges(|_| {}, base);
    }

    /// Stores all triangles for this mesh in the destination vector.
    pub fn triangles(&self, dst: &mut Vec<[usize; 3]>) {
        for f in &self.f {
            dst.extend(f.as_triangle_fan());
        }
    }

    /// Normalize this mesh's geometry to lay within [-1, 1].
    /// Outputs scale and translation to reposition back to the original dimension.
    pub fn normalize(&mut self) -> (F, [F; 3]) {
        normalize_transform(&mut self.v)
    }
    pub fn normalize_colors(&mut self) -> (F, [F; 3]) {
        normalize_transform(&mut self.vert_colors)
    }
    /// Given a scale and translation output from normalization, reset the geometry to its
    /// original position.
    pub fn denormalize(&mut self, scale: F, trans: [F; 3]) {
        denormalize_transform(&mut self.v, scale, trans)
    }
    pub fn denormalize_colors(&mut self, scale: F, trans: [F; 3]) {
        denormalize_transform(&mut self.vert_colors, scale, trans)
    }
    /// After flattening a scene into a mesh, repopulate the original scene with a modified
    /// flattened mesh. (Opposite of flattening, unflattening, inflating).
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
            f.remap(|flat_vi| {
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

                    if let Some(&vc) = self.vert_colors.get(flat_vi) {
                        mesh.vert_colors.push(vc);
                        assert_eq!(mesh.v.len(), mesh.vert_colors.len());
                    }

                    macro_rules! opt_va_push {
                        ($k: ident) => {{
                            if let Some(&v) = self.vertex_attrs.$k.get(flat_vi) {
                                mesh.vertex_attrs.$k.push(v);
                                assert_eq!(mesh.v.len(), mesh.vertex_attrs.$k.len());
                            }
                        }};
                    }

                    opt_va_push!(height);
                    opt_va_push!(tangent);
                    opt_va_push!(bitangent);

                    opt_va_push!(opacity);
                    opt_va_push!(scale);
                    opt_va_push!(rot);
                    opt_va_push!(sph_harmonic_coeff);

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
    /// Note that vertices at bitwise identical positions will be unified.
    /// The face order will be preserved exactly with the input mesh.
    pub fn geometry_only(&mut self) -> HashMap<[U; 3], usize> {
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
            face_mesh_idx: std::mem::take(&mut self.face_mesh_idx),
            face_mat_idx: std::mem::take(&mut self.face_mat_idx),
            ..Default::default()
        };

        new_map
    }

    pub fn copy_attribs<const N: usize>(
        &mut self,
        og_v: &[[F; 3]],
        new_vert_map: HashMap<[U; 3], usize>,
        og_attribs: &[[F; N]],
    ) -> Vec<[F; N]> {
        let mut new = vec![[0.; N]; self.v.len()];
        let mut ws = vec![0; self.v.len()];
        for (vi, v) in og_v.iter().enumerate() {
            let new_idx = new_vert_map[&v.map(F::to_bits)];
            let old_val = og_attribs[vi];
            new[new_idx] = add(new[new_idx], old_val);
            ws[new_idx] += 1;
        }
        for (vi, val) in new.iter_mut().enumerate() {
            debug_assert_ne!(ws[vi], 0);
            *val = divk(*val, ws[vi] as F);
        }
        new
    }

    pub fn clear_vertex_normals(&mut self) {
        if self.n.is_empty() {
            return;
        }

        let v = std::mem::take(&mut self.v);
        let uv = std::mem::take(&mut self.uv);

        let mut new_v = vec![];
        let mut new_uv: [Vec<[F; 2]>; MAX_UV] = std::array::from_fn(|_| vec![]);

        // original vertex position to new vertex index
        let mut new_map: HashMap<_, usize> = HashMap::new();

        let key = |vi: usize| {
            let vk = v[vi].map(F::to_bits);
            let uv_k: [_; MAX_UV] =
                std::array::from_fn(|i| uv[i].get(vi).map(|uv| uv.map(F::to_bits)));
            (vk, uv_k)
        };

        for f in self.f.iter_mut() {
            for vi in f.as_mut_slice() {
                let k = key(*vi);
                *vi = match new_map.entry(k) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(vac) => {
                        let new_idx = new_v.len();
                        new_v.push(v[*vi]);
                        for i in 0..MAX_UV {
                            let Some(&uv) = uv[i].get(*vi) else {
                                continue;
                            };
                            new_uv[i].push(uv);
                        }
                        vac.insert(new_idx);
                        new_idx
                    }
                }
            }
        }

        *self = Mesh {
            v: new_v,
            uv: new_uv,
            f: std::mem::take(&mut self.f),
            face_mesh_idx: std::mem::take(&mut self.face_mesh_idx),
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
    /// For all values from the other mesh, append them to this mesh.
    pub fn append(&mut self, o: &mut Self) {
        let vert_offset = self.v.len();
        self.v.append(&mut o.v);
        self.n.append(&mut o.n);
        for i in 0..MAX_UV {
            self.uv[i].append(&mut o.uv[i]);
        }
        self.vert_colors.append(&mut o.vert_colors);
        for f in o.f.iter_mut() {
            for vi in f.as_mut_slice() {
                *vi += vert_offset
            }
        }
        self.f.append(&mut o.f);
        self.face_mesh_idx.append(&mut o.face_mesh_idx);
        self.face_mat_idx.append(&mut o.face_mat_idx);
        self.joint_idxs.append(&mut o.joint_idxs);
        self.joint_weights.append(&mut o.joint_weights);
    }

    /// Computes the number of vertices used in this mesh O(|V|).
    pub fn num_used_vertices(&self) -> usize {
        let mut used: BTreeSet<usize> = BTreeSet::new();
        for f in &self.f {
            used.extend(f.as_slice().iter());
        }
        used.len()
    }

    /// Deletes vertices present in this mesh, but which are not used in any faces.
    /// Also returns the mapping from original vertices -> new_vertex, where usize::MAX
    /// indicates no mapping.
    pub fn delete_unused_vertices(&mut self) -> (usize, Vec<usize>) {
        let curr_len = self.v.len();
        // either use btreeset or vec here, but must remain in sorted order
        let mut used: Vec<usize> = self
            .f
            .iter()
            .flat_map(|f| f.as_slice().iter().copied())
            .collect();
        used.sort_unstable();
        used.dedup();
        let num_used = used.len();

        let mut remap = vec![usize::MAX; self.v.len()];
        for (new_vi, og_vi) in used.into_iter().enumerate() {
            *unsafe { remap.get_unchecked_mut(og_vi) } = new_vi;
        }
        let remap = remap;
        for f in &mut self.f {
            f.remap(|vi| unsafe { *remap.get_unchecked(vi) });
        }

        macro_rules! clear_vec {
            ($vec: expr) => {{
                let mut i = 0;
                $vec.retain(|_| {
                    let keep = unsafe { *remap.get_unchecked(i) } != usize::MAX;
                    i += 1;
                    keep
                });
            }};
        }

        clear_vec!(&mut self.v);
        assert_eq!(self.v.len(), num_used);

        clear_vec!(&mut self.n);
        for i in 0..MAX_UV {
            clear_vec!(&mut self.uv[i]);
        }
        clear_vec!(&mut self.joint_idxs);
        clear_vec!(&mut self.joint_weights);
        clear_vec!(&mut self.vert_colors);

        (curr_len - num_used, remap)
    }

    /// Deletes empty faces from this mesh.
    pub fn delete_empty_faces(&mut self) {
        let mut fi = 0;
        while fi < self.f.len() {
            if !self.f[fi].is_empty() {
                fi += 1;
                continue;
            }
            self.f.swap_remove(fi);
            if fi < self.face_mesh_idx.len() {
                self.face_mesh_idx.swap_remove(fi);
            }
            if fi < self.face_mat_idx.len() {
                self.face_mat_idx.swap_remove(fi);
            }
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

/// Normalizes a vector of N-dim arrays so that they fit in [-1; N] to [1; N], with uniform
/// scaling. Returns the transformation and scale applied.
pub fn normalize_transform<const N: usize>(v: &mut [[F; N]]) -> (F, [F; N]) {
    // Normalize the geometry of this mesh to lay in the unit box.
    let [l, h] = v
        .iter()
        .fold([[F::INFINITY; N], [F::NEG_INFINITY; N]], |[l, h], n| {
            [from_fn(|i| l[i].min(n[i])), from_fn(|i| h[i].max(n[i]))]
        });
    let center = kmul(0.5, add(l, h));
    for v in v.iter_mut() {
        *v = sub(*v, center);
    }
    let largest_val = v
        .iter()
        .fold(0., |m, v| v.iter().fold(m, |m, c| c.abs().max(m)));
    let scale = if largest_val == 0. {
        1.
    } else {
        largest_val.recip()
    };
    for v in v.iter_mut() {
        *v = kmul(scale, *v);
    }
    (scale, center)
}

/// Returns a scaled and transformed set of N-dim arrays with a given scale and transformation
/// back to their original position.
pub fn denormalize_transform<const N: usize>(v: &mut [[F; N]], scale: F, trans: [F; N]) {
    assert_ne!(scale, 0.);
    let inv_scale = scale.recip();
    for v in v {
        *v = add(kmul(inv_scale, *v), trans);
    }
}
