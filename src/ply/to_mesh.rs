use super::Ply;
use crate::F;
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
        let vert_colors = vc
            .into_iter()
            .map(|rgb| rgb.map(|v| v as F / 255.))
            .collect::<Vec<_>>();
        Mesh {
            v,
            f,
            n,
            vert_colors,

            uv: [uv, vec![], vec![], vec![]],
            face_mesh_idx: vec![],
            face_mat_idx: vec![],
            joint_idxs: vec![],
            joint_weights: vec![],

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
        let vc = vc
            .into_iter()
            .map(|c| c.map(|v| (v.clamp(0., 1.) * 255.) as u8))
            .collect::<Vec<_>>();
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
