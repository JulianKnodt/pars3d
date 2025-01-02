use super::{STLFace, STL};
use crate::mesh::Mesh;
use crate::{add, cross, normalize, sub, FaceKind};
use std::array::from_fn;

impl From<STL> for Mesh {
    fn from(stl: STL) -> Self {
        let STL { name, faces } = stl;

        let mut v = vec![];
        let mut n = vec![];
        let mut f = vec![];
        for face in faces {
            let curr_vi = v.len();
            for i in 0..3 {
                n.push(face.normal);
                v.push(face.pos[i]);
            }
            f.push(FaceKind::Tri(from_fn(|i| curr_vi + i)));
        }

        Mesh {
            v,
            f,
            n,
            name,

            uv: from_fn(|_| vec![]),
            face_mesh_idx: vec![],
            vert_colors: vec![],
            face_mat_idx: vec![],
            joint_idxs: vec![],
            joint_weights: vec![],
        }
    }
}

impl From<Mesh> for STL {
    fn from(mesh: Mesh) -> Self {
        let Mesh { v, f, n, name, .. } = mesh;
        let mut faces = vec![];
        for f in &f {
            for t in f.as_triangle_fan() {
                let [v0, v1, v2] = t.map(|i| v[i]);
                let normal = if t.iter().all(|&i| i < n.len()) {
                    normalize(cross(sub(v2, v0), sub(v1, v0)))
                } else {
                    normalize(t.map(|i| n[i]).into_iter().fold([0.; 3], add))
                };
                faces.push(STLFace {
                    normal,
                    pos: [v0, v1, v2],
                });
            }
        }
        STL { name, faces }
    }
}
