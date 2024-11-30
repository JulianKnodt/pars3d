use crate::{FaceKind, F};

pub mod export;
pub mod parser;

/// From/Into conversions between unified representation and FBX representation.
pub mod to_mesh;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    meshes: Vec<FBXMesh>,
    nodes: Vec<FBXNode>,

    root_nodes: Vec<usize>,

    global_settings: FBXSettings,
}

impl FBXScene {
    pub(crate) fn parent_node(&self, node: usize) -> Option<usize> {
        self.nodes.iter().position(|n| n.children.contains(&node))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXNode {
    id: usize,
    mesh: Option<usize>,
    children: Vec<usize>,
    name: String,
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

    global_mat: Option<usize>,
    per_face_mat: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FBXSettings {
    up_axis: i32,
    up_axis_sign: i32,
    front_axis: i32,
    front_axis_sign: i32,
    coord_axis: i32,
    coord_axis_sign: i32,
    og_up_axis: i32,
    og_up_axis_sign: i32,

    unit_scale_factor: f64,
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
