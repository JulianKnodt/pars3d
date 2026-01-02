use super::Ply;
use crate::mesh::Mesh;

impl From<Ply> for Mesh {
    fn from(ply: Ply) -> Self {
        let Ply {
            v,
            f,
            n,
            uv,
            vc,
            vertex_attrs,
        } = ply;
        Mesh {
            v,
            f,
            n,
            vert_colors: vc,

            uv: [uv, vec![], vec![], vec![]],
            face_mesh_idx: vec![],
            face_mat_idx: vec![],
            joint_idxs: vec![],
            joint_weights: vec![],
            l: vec![],

            name: String::new(),
            vertex_attrs,
        }
    }
}

impl From<Mesh> for Ply {
    fn from(mesh: Mesh) -> Self {
        let Mesh {
            v,
            f,
            n,
            vert_colors: vc,
            uv: [uv, _, _, _],
            vertex_attrs,
            ..
        } = mesh;

        Ply {
            v,
            f,
            vc,
            uv,
            n,
            vertex_attrs,
        }
    }
}
