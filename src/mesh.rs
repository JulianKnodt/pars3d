use super::obj::ObjObject;
use super::F;

#[cfg(feature = "gltf")]
use super::gltf::GLTFMesh;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceKind {
    Tri([usize; 3]),
    Quad([usize; 4]),
    Poly(Vec<usize>),
}

impl FaceKind {
    pub fn as_slice(&self) -> &[usize] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_slice(),
            Quad(q) => q.as_slice(),
            Poly(v) => v.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_mut_slice(),
            Quad(q) => q.as_mut_slice(),
            Poly(v) => v.as_mut_slice(),
        }
    }
    pub fn len(&self) -> usize {
        use FaceKind::*;
        match self {
            Tri(_) => 3,
            Quad(_) => 4,
            Poly(v) => v.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub v: Vec<[F; 3]>,
    pub f: Vec<FaceKind>,

    /// 1-1 relation between vertices and joint/idxs weights.
    pub joint_idxs: Vec<[u16; 4]>,
    pub joint_weights: Vec<[F; 4]>,
}

impl From<ObjObject> for Mesh {
    fn from(obj: ObjObject) -> Self {
        let mut f = vec![];

        let iter = obj.f.into_iter().filter_map(|f| match f.v.as_slice() {
            [] | [_] | [_, _] => None,
            &[a, b, c] => Some(FaceKind::Tri([a, b, c])),
            &[a, b, c, d] => Some(FaceKind::Quad([a, b, c, d])),
            _ => Some(FaceKind::Poly(f.v)),
        });
        f.extend(iter);

        Self {
            v: obj.v,
            f,

            joint_idxs: vec![],
            joint_weights: vec![],
        }
    }
}

#[cfg(feature = "gltf")]
impl From<GLTFMesh> for Mesh {
    fn from(gltf_mesh: GLTFMesh) -> Self {
        if !gltf_mesh.joint_idxs.is_empty() {
            assert_eq!(gltf_mesh.v.len(), gltf_mesh.joint_idxs.len());
        }
        let f = gltf_mesh
            .f
            .into_iter()
            .map(|f| FaceKind::Tri(f))
            .collect::<Vec<_>>();

        Self {
            v: gltf_mesh.v,
            f,

            joint_idxs: gltf_mesh.joint_idxs,
            joint_weights: gltf_mesh.joint_weights,
        }
    }
}
