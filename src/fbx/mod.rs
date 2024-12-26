use crate::mesh::DecomposedTransform;
use crate::{FaceKind, F};
use std::sync::atomic::AtomicUsize;

pub mod export;
pub mod parser;

/// From/Into conversions between unified representation and FBX representation.
pub mod to_mesh;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    id: usize,
    pub meshes: Vec<FBXMesh>,
    pub nodes: Vec<FBXNode>,

    materials: Vec<FBXMaterial>,

    skins: Vec<FBXSkin>,

    anims: Vec<FBXAnim>,

    root_nodes: Vec<usize>,

    pub global_settings: FBXSettings,

    file_id: Vec<u8>,
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
    by_id_or_new!(skin_by_id_or_new, skins);
    by_id_or_new!(anim_by_id_or_new, anims);
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXNode {
    id: usize,
    mesh: Option<usize>,
    // also store materials used in each node
    materials: Vec<usize>,

    children: Vec<usize>,
    name: String,

    transform: DecomposedTransform,

    skin: Option<usize>,
}

impl FBXNode {
    pub fn is_limb_node(&self) -> bool {
        todo!();
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMaterial {
    id: usize,
    name: String,
    diffuse_color: [F; 3],
    specular_color: [F; 3],
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

    // indices into what?
    indices: Vec<usize>,
    weights: Vec<F>,

    name: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FBXAnim {
    id: usize,

    times: Vec<u32>,
    values: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMesh {
    id: usize,
    name: String,

    pub v: Vec<[F; 3]>,
    pub f: Vec<FaceKind>,

    pub n: VertexAttribute<3>,
    pub uv: VertexAttribute<2>,
    color: VertexAttribute<3>,

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
    pub fn v(&self, vi: usize) -> [T; N]
    where
        T: Copy,
    {
        match self.ref_kind {
            RefKind::Direct => self.values[vi],
            RefKind::IndexToDirect => {
                assert!(!self.indices.is_empty());
                self.values[self.indices[vi]]
            }
        }
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
