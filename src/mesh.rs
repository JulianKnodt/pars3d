use super::obj::ObjObject;
use super::F;

use super::FaceKind;

use std::array::from_fn;
use std::collections::HashMap;

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

impl Mesh {
    pub fn flip_uv_v(&mut self) {
        for uv_chan in self.uv.iter_mut() {
            for uvs in uv_chan.iter_mut() {
                uvs[1] = 1. - uvs[1];
            }
        }
    }
}

impl From<ObjObject> for Mesh {
    fn from(obj: ObjObject) -> Self {
        let mut fs = vec![];

        let mut verts = HashMap::new();
        let mut v = vec![];
        let mut uv = vec![];
        let mut n = vec![];

        for f in obj.f.into_iter() {
            if f.v.len() < 3 {
                continue;
            }
            macro_rules! key_i {
                ($i: expr) => {
                    (f.v[$i], f.vt.get($i).copied(), f.vn.get($i).copied())
                };
            }
            for i in 0..f.v.len() {
                let key = key_i!(i);
                if !verts.contains_key(&key) {
                    v.push(obj.v[f.v[i]]);
                    if let Some(vt) = key.1 {
                        uv.push(obj.vt[vt]);
                    };
                    if let Some(vn) = key.2 {
                        n.push(obj.vn[vn]);
                    };
                    verts.insert(key, verts.len());
                }
            }
            let f = match f.v.len() {
                0 | 1 | 2 => unreachable!(),
                3 => FaceKind::Tri(from_fn(|i| verts[&key_i!(i)])),
                4 => FaceKind::Quad(from_fn(|i| verts[&key_i!(i)])),
                n => FaceKind::Poly((0..n).map(|i| verts[&key_i!(i)]).collect::<Vec<_>>()),
            };
            fs.push(f);
        }

        // TODO here would need to dedup vertices and UVs so that each index is unique.
        Self {
            v,
            f: fs,
            n,
            uv: [uv, vec![], vec![], vec![]],

            joint_idxs: vec![],
            joint_weights: vec![],
        }
    }
}

#[cfg(feature = "gltf")]
use super::gltf::GLTFMesh;

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
