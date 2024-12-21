use super::Ply;
use crate::mesh::Mesh;
use std::array::from_fn;

impl From<Ply> for Mesh {
    fn from(ply: Ply) -> Self {
        let Ply { v, f, vc: _ } = ply;
        Mesh {
            v,
            f,

            uv: from_fn(|_| vec![]),
            face_mesh_idx: vec![],
            face_mat_idx: vec![],
            n: vec![],
            joint_idxs: vec![],
            joint_weights: vec![],
            name: String::new(),
        }
    }
}

impl From<Mesh> for Ply {
    fn from(mesh: Mesh) -> Self {
        let Mesh { v, f, .. } = mesh;
        Ply { v, f, vc: vec![] }
    }
}
