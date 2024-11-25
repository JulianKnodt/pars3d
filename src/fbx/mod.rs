use crate::{FaceKind, F};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    meshes: Vec<FBXMesh>,
    nodes: Vec<FBXNode>,

    root_nodes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXNode {
    id: usize,
    mesh: Option<usize>,
    children: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMesh {
    id: usize,

    v: Vec<[F; 3]>,
    f: Vec<FaceKind>,

    n: Vec<[F; 3]>,
    // for each vertex, what is its normal
    vert_norm_idx: Vec<usize>,

    uv: Vec<[F; 2]>,
    uv_idx: Vec<usize>,

    global_mat: Option<usize>,
    per_face_mat: Vec<usize>,
}

pub mod parser;
