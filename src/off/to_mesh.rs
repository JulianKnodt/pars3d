use super::OFF;
use crate::mesh::Mesh;

impl From<OFF> for Mesh {
    fn from(off: OFF) -> Self {
        let OFF { v, f } = off;
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
        OFF { v, f }
    }
}
