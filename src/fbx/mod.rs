use crate::mesh::DecomposedTransform;
use crate::{FaceKind, F};
use std::sync::atomic::AtomicUsize;

pub mod export;
pub mod parser;

/// From/Into conversions between unified representation and FBX representation.
pub mod to_mesh;

/// From/Into conversions between animations and FBX representation.
pub mod to_anim;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    id: usize,
    pub meshes: Vec<FBXMesh>,
    pub nodes: Vec<FBXNode>,

    materials: Vec<FBXMaterial>,
    textures: Vec<FBXTexture>,

    pub skins: Vec<FBXSkin>,
    pub clusters: Vec<FBXCluster>,
    pub poses: Vec<FBXPose>,

    pub anim_stacks: Vec<FBXAnimStack>,
    pub anim_layers: Vec<FBXAnimLayer>,
    pub anim_curves: Vec<FBXAnimCurve>,
    pub anim_curve_nodes: Vec<FBXAnimCurveNode>,

    pub blendshapes: Vec<FBXBlendshape>,
    pub blendshape_channels: Vec<FBXBlendshapeChannel>,

    root_nodes: Vec<usize>,

    pub global_settings: FBXSettings,

    file_id: [u8; 16],
}

macro_rules! by_id_or_new {
    ($fn_name: ident, $field: ident) => {
        pub fn $fn_name(&mut self, id: usize) -> usize {
            if let Some(i) = self.$field.iter().position(|v| v.id == id) {
                return i;
            };

            self.$field.push(Default::default());
            self.$field.last_mut().unwrap().id = id;
            self.$field.len() - 1
        }
    };
}

impl FBXScene {
    pub(crate) fn parent_node(&self, node: usize) -> Option<usize> {
        self.nodes.iter().position(|n| n.children.contains(&node))
    }

    by_id_or_new!(mat_by_id_or_new, materials);
    by_id_or_new!(mesh_by_id_or_new, meshes);
    by_id_or_new!(node_by_id_or_new, nodes);
    by_id_or_new!(cluster_by_id_or_new, clusters);
    by_id_or_new!(anim_curve_by_id_or_new, anim_curves);
    by_id_or_new!(anim_curve_node_by_id_or_new, anim_curve_nodes);
    by_id_or_new!(blendshape_by_id_or_new, blendshapes);
    by_id_or_new!(texture_by_id_or_new, textures);
    by_id_or_new!(pose_by_id_or_new, poses);
    by_id_or_new!(skin_by_id_or_new, skins);
    by_id_or_new!(blendshape_channel_by_id_or_new, blendshape_channels);
    by_id_or_new!(anim_layer_by_id_or_new, anim_layers);
    by_id_or_new!(anim_stack_by_id_or_new, anim_stacks);

    pub fn id_kind(&self, id: usize) -> FieldKind {
        macro_rules! check {
            ($items: expr, $kind: expr) => {
                let has_id = $items.iter().find(|v| v.id == id).is_some();
                if has_id {
                    return $kind;
                }
            };
        }

        check!(self.materials, FieldKind::Material);
        check!(self.meshes, FieldKind::Mesh);
        check!(self.nodes, FieldKind::Node);
        check!(self.clusters, FieldKind::Cluster);
        check!(self.anim_curves, FieldKind::AnimCurve);
        check!(self.blendshapes, FieldKind::Blendshape);
        check!(self.textures, FieldKind::Texture);
        check!(self.poses, FieldKind::Pose);
        check!(self.skins, FieldKind::Skin);
        check!(self.blendshape_channels, FieldKind::BlendshapeChannel);
        check!(self.anim_layers, FieldKind::AnimLayer);
        check!(self.anim_curve_nodes, FieldKind::AnimCurveNode);
        check!(self.anim_stacks, FieldKind::AnimStack);

        FieldKind::Unknown
    }

    /// Number of animations in this FBX
    pub fn num_animations(&self) -> usize {
        self.anim_layers.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Material,
    Mesh,
    Node,
    Cluster,
    AnimCurve,
    AnimCurveNode,
    Blendshape,
    Texture,
    Pose,
    Skin,
    BlendshapeChannel,
    AnimLayer,
    AnimStack,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXNode {
    pub id: usize,
    pub mesh: Option<usize>,
    // also store materials used in each node
    materials: Vec<usize>,

    children: Vec<usize>,
    parent: Option<usize>,
    name: String,

    transform: DecomposedTransform,

    hidden: bool,

    cluster: Option<usize>,

    pub limb_node_id: Option<usize>,
    pub is_null_node: bool,
}

impl FBXNode {
    pub fn is_limb_node(&self) -> bool {
        self.limb_node_id.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMaterial {
    id: usize,
    name: String,
    diffuse_color: [F; 3],
    specular_color: [F; 3],
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXTexture {
    id: usize,
    name: String,
    file_name: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum FBXMeshMaterial {
    #[default]
    None,

    Global(usize),
    PerFace(Vec<usize>),
}

impl FBXMeshMaterial {
    pub fn as_slice(&self) -> &[usize] {
        use FBXMeshMaterial::*;
        match self {
            None => &[],
            Global(v) => std::slice::from_ref(v),
            PerFace(vs) => vs.as_slice(),
        }
    }
    pub(crate) fn remap(&mut self, map: impl Fn(usize) -> usize) {
        use FBXMeshMaterial::*;
        match self {
            None => {}
            Global(v) => *v = map(*v),
            PerFace(vs) => {
                for v in vs {
                    *v = map(*v);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXSkin {
    id: usize,
    deform_acc: F,
    clusters: Vec<usize>,
    mesh: usize,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXCluster {
    id: usize,

    // indices into what? (might be poses?)
    indices: Vec<usize>,
    weights: Vec<F>,

    tform: [[F; 4]; 4],
    tform_link: [[F; 4]; 4],
    tform_assoc_model: [[F; 4]; 4],

    // Note that this is not nullable, every cluster should be associated with a node
    node: usize,
    // This should be 1 Skin to N clusters.
    skin: usize,

    name: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXPose {
    id: usize,
    // id of node
    nodes: Vec<usize>,
    matrices: Vec<[[F; 4]; 4]>,

    name: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXBlendshape {
    id: usize,

    name: String,
    indices: Vec<usize>,
    v: Vec<[F; 3]>,
    n: Vec<[F; 3]>,

    channels: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXBlendshapeChannel {
    id: usize,

    deform_percent: F,
    full_weights: Vec<F>,
    mesh: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FBXAnimStack {
    id: usize,
    local_start: i64,
    local_stop: i64,
    ref_start: i64,
    ref_stop: i64,
    name: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AnimCurveNodeKey {
    X,
    Y,
    Z,
    DeformPercent,
    #[default]
    None,
}

impl AnimCurveNodeKey {
    fn from_str(s: &str) -> Self {
        match s {
            "d|X" => Self::X,
            "d|Y" => Self::Y,
            "d|Z" => Self::Z,
            "d|DeformPercent" => Self::DeformPercent,
            x => panic!("{x}"),
        }
    }
    fn as_str(&self) -> &'static str {
        use AnimCurveNodeKey::*;
        match self {
            X => "d|X",
            Y => "d|Y",
            Z => "d|Z",
            DeformPercent => "d|DeformPercent",
            None => unreachable!("Should've set AnimCurveNodeKey"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FBXAnimCurve {
    id: usize,

    default: F,

    /// needs to be u64 otherwise not enough bits (that's crazy).
    times: Vec<u64>,

    values: Vec<F>,

    flags: Vec<i32>,
    data: Vec<F>,
    ref_count: Vec<i32>,

    anim_curve_node: usize,
    anim_curve_node_key: AnimCurveNodeKey,

    name: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NodeAnimAttrKey {
    Translation,
    Rotation,
    Scaling,
    // Technically a blend shape animation not node
    DeformPercent,
    #[default]
    None,
}

impl NodeAnimAttrKey {
    fn from_str(s: &str) -> Self {
        match s {
            "Lcl Translation" => Self::Translation,
            "Lcl Rotation" => Self::Rotation,
            "Lcl Scaling" => Self::Scaling,
            "DeformPercent" => Self::DeformPercent,
            x => panic!("{x}"),
        }
    }
    fn as_str(&self) -> &'static str {
        use NodeAnimAttrKey::*;
        match self {
            Translation => "Lcl Translation",
            Rotation => "LcL Rotation",
            Scaling => "Lcl Scaling",
            DeformPercent => "DeformPercent",
            None => unreachable!("Should've set NodeAnimAttrKey"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshOrNode {
    #[default]
    None,
    Node(usize),
    Mesh(usize),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FBXAnimCurveNode {
    id: usize,

    dx: Option<F>,
    dy: Option<F>,
    dz: Option<F>,
    deform_percent: Option<F>,

    // I think only one of these can be set, but not sure?
    layer: usize,

    // What is this anim curve node related to?
    // assumes that each anim curve node related to one thing.
    rel: MeshOrNode,
    rel_key: NodeAnimAttrKey,

    name: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FBXAnimLayer {
    id: usize,

    // TODO may need to check this is 1-1
    anim_stack: usize,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMesh {
    id: usize,
    pub name: String,

    pub v: Vec<[F; 3]>,
    pub f: Vec<FaceKind>,

    pub n: VertexAttribute<3>,
    pub uv: VertexAttribute<2>,
    pub color: VertexAttribute<3>,

    skin: Option<usize>,

    mat: FBXMeshMaterial,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VertexAttribute<const N: usize, T = F> {
    values: Vec<[T; N]>,
    indices: Vec<usize>,

    ref_kind: RefKind,
    map_kind: VertexMappingKind,
}

impl<T, const N: usize> VertexAttribute<N, T> {
    pub fn len(&self) -> usize {
        match self.ref_kind {
            RefKind::Direct => self.values.len(),
            RefKind::IndexToDirect => self.indices.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn v(&self, wedge_idx: usize, vi: usize) -> Option<[T; N]>
    where
        T: Copy,
    {
        let idx = match self.map_kind {
            VertexMappingKind::ByVertices => vi,
            VertexMappingKind::Wedge => wedge_idx,
        };
        let v = match self.ref_kind {
            RefKind::Direct => self.values.get(idx)?,
            RefKind::IndexToDirect => {
                assert!(!self.indices.is_empty());
                self.values.get(self.indices[idx])?
            }
        };
        Some(*v)
    }

    /// Whether indices are required for this vertex attribute
    pub fn requires_indices(&self) -> bool {
        !(self.map_kind.is_by_vertices() && self.ref_kind.is_direct())
    }
}

/// How to map some information to vertices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefKind {
    #[default]
    Direct,
    IndexToDirect,
}

impl RefKind {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Direct" => Self::Direct,
            "IndexToDirect" => Self::IndexToDirect,
            _ => todo!("Unknown ref info type {s}, please file a bug."),
        }
    }
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Direct => "Direct",
            Self::IndexToDirect => "IndexToDirect",
        }
    }
    pub fn is_direct(&self) -> bool {
        *self == Self::Direct
    }
}

/// How to map some information to a mesh
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VertexMappingKind {
    /// 1-1 mapping with vertices
    #[default]
    ByVertices,
    /// Wedge (Per face has different values)
    Wedge,
}

impl VertexMappingKind {
    pub fn from_str(s: &str) -> Self {
        match s {
            "ByPolygonVertex" => VertexMappingKind::Wedge,
            "ByVertice" => VertexMappingKind::ByVertices,
            _ => todo!("Unknown vertex mapping kind {s}, please file a bug."),
        }
    }
    pub fn to_str(&self) -> &'static str {
        match self {
            VertexMappingKind::Wedge => "ByPolygonVertex",
            VertexMappingKind::ByVertices => "ByVertice",
        }
    }
    pub fn is_by_vertices(&self) -> bool {
        *self == VertexMappingKind::ByVertices
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FBXSettings {
    up_axis: i32,
    up_axis_sign: i32,
    front_axis: i32,
    front_axis_sign: i32,
    coord_axis: i32,
    coord_axis_sign: i32,
    // no idea wtf these next two are
    og_up_axis: i32,
    og_up_axis_sign: i32,

    unit_scale_factor: f64,
    // no idea wtf this is
    og_unit_scale_factor: f64,

    time_span_start: i64,
    time_span_stop: i64,

    frame_rate: f64,
}

impl Default for FBXSettings {
    fn default() -> Self {
        FBXSettings {
            up_axis: 1,
            up_axis_sign: 1,
            front_axis: 0,
            front_axis_sign: 1,
            coord_axis: 2,
            coord_axis_sign: 1,
            og_up_axis: 1,
            og_up_axis_sign: 1,

            unit_scale_factor: 1.,
            og_unit_scale_factor: 1.,

            time_span_start: 0,
            time_span_stop: 0,

            frame_rate: 0.,
        }
    }
}

/// Construct an ID for usage in FBX.
/// Guaranteed to be unique, but may overflow if left running for too long.
pub(crate) fn id() -> usize {
    static mut CURR_ID: AtomicUsize = AtomicUsize::new(3333);
    let id = unsafe {
        #[allow(static_mut_refs)]
        CURR_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    };
    assert_ne!(id, 0);
    id
}
