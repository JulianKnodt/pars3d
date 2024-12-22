use super::OFF;
use crate::mesh::Mesh;
use crate::FaceKind;

impl From<OFF> for Mesh {
    fn from(off: OFF) -> Self {
        let OFF { v, f } = off;
        let f = f.into_iter().map(FaceKind::Tri).collect::<Vec<_>>();
        Mesh {
            v,
            f,
            ..Default::default()
        }
    }
}

impl From<Mesh> for OFF {
    fn from(mesh: Mesh) -> Self {
        let mut new_f = Vec::with_capacity(mesh.num_tris());
        let Mesh { v, f, .. } = mesh;
        for f in f.into_iter() {
            for t in f.as_triangle_fan() {
                new_f.push(t);
            }
        }

        OFF { v, f: new_f }
    }
}
