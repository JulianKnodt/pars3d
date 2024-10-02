use super::obj::ObjObject;
use super::F;

use super::FaceKind;

#[cfg(feature = "gltf")]
use super::gltf::GLTFMesh;

const MAX_UV: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub v: Vec<[F; 3]>,
    pub uv: [Vec<[F; 2]>; MAX_UV],

    pub n: Vec<[F; 3]>,
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

        // TODO here would need to dedup vertices and UVs so that each index is unique.
        Self {
            v: obj.v,
            f,
            n: obj.vn,
            uv: [obj.vt, vec![], vec![], vec![]],

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
            .map(FaceKind::Tri)
            .collect::<Vec<_>>();

        Self {
            v: gltf_mesh.v,
            f,
            n: vec![],
            uv: [gltf_mesh.uvs, vec![], vec![], vec![]],

            joint_idxs: gltf_mesh.joint_idxs,
            joint_weights: gltf_mesh.joint_weights,
        }
    }
}
