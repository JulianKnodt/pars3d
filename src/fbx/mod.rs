use crate::mesh::DecomposedTransform;
use crate::{FaceKind, F};
use std::sync::atomic::AtomicUsize;

pub mod export;
pub mod parser;

/// From/Into conversions between unified representation and FBX representation.
pub mod to_mesh;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    meshes: Vec<FBXMesh>,
    nodes: Vec<FBXNode>,

    materials: Vec<FBXMaterial>,

    skins: Vec<FBXSkin>,

    anims: Vec<FBXAnim>,

    root_nodes: Vec<usize>,

    global_settings: FBXSettings,

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

    v: Vec<[F; 3]>,
    f: Vec<FaceKind>,

    n: Vec<[F; 3]>,
    // for each vertex, what is its normal
    vert_norm_idx: Vec<usize>,

    // TODO need to add multiple channels here
    uv: Vec<[F; 2]>,
    uv_idx: Vec<usize>,

    vertex_colors: Vec<[F; 3]>,
    vertex_color_idx: Vec<usize>,

    mat: FBXMeshMaterial,
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
