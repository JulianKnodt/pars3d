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
