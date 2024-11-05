use crate::{FaceKind, F};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXScene {
    meshes: Vec<FBXMesh>,
    nodes: Vec<FBXNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FBXNode {
    mesh: Option<usize>,
    children: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FBXMesh {
    v: Vec<[F; 3]>,
    f: Vec<FaceKind>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FBXObject {
    id: i64,
}

pub mod parser;
