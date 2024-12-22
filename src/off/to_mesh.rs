use super::OFF;
use crate::mesh::Mesh;
use std::array::from_fn;
use crate::FaceKind;

impl From<Off> for Mesh {
    fn from(off: OFF) -> Self {
        let OFF { v, f } = off;
        let f = f.into_iter.map(FaceKind::Tri).collect::<Vec<_>>()
        Mesh {
            v,
            f,
            ..Default::default()
        }
    }
}

impl From<Mesh> for OFF {
    fn from(mesh: Mesh) -> Self {
        let Mesh { v, f, .. } = mesh;
        let f = f.into_iter().flat_map(|f| f.as_triangle_fan()).collect::<Vec<_>>()
        OFF { v, f }
    }
}
