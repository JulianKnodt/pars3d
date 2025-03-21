use super::Ply;
use crate::mesh::Mesh;
use crate::F;
use std::array::from_fn;

impl From<Ply> for Mesh {
    fn from(ply: Ply) -> Self {
        let Ply { v, f, vc } = ply;
        let vert_colors = vc
            .into_iter()
            .map(|rgb| rgb.map(|v| v as F / 255.))
            .collect::<Vec<_>>();
        Mesh {
            v,
            f,
            vert_colors,

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
        let Mesh {
            v,
            f,
            vert_colors: vc,
            ..
        } = mesh;
        let vc = vc
            .into_iter()
            .map(|c| c.map(|v| (v.clamp(0., 1.) * 255.) as u8))
            .collect::<Vec<_>>();
        Ply { v, f, vc }
    }
}
