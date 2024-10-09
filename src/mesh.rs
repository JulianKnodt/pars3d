use super::obj::{Obj, ObjObject};
use super::{add, kmul, sub, FaceKind, F};

use std::array::from_fn;
use std::collections::HashMap;

const MAX_UV: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    Diffuse,
    Normal,
    Emissive,
    Specular,
    Metallic,
    Roughness,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Material {
    textures: Vec<(TextureKind, image::DynamicImage)>,
    name: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Node {
    pub mesh: Option<usize>,
    pub children: Vec<usize>,
    // TODO more fields
    pub transform: [[F; 4]; 4],
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Scene {
    pub root_nodes: Vec<usize>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Mesh {
    pub v: Vec<[F; 3]>,
    pub uv: [Vec<[F; 2]>; MAX_UV],

    pub n: Vec<[F; 3]>,

    pub f: Vec<FaceKind>,
    /// Which mesh did this face come from?
    /// Used when flattening a scene into a single mesh.
    pub face_mesh_idx: Vec<usize>,

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
    /// Normalize this mesh's geometry to lay within [-1, 1].
    /// Outputs scale and translation to reposition back to the original dimension.
    pub fn normalize(&mut self) -> (F, [F; 3]) {
        // Normalize the geometry of this mesh to lay in the unit box.
        let [l, h] = self
            .v
            .iter()
            .fold([[F::INFINITY; 3], [F::NEG_INFINITY; 3]], |[l, h], n| {
                [from_fn(|i| l[i].min(n[i])), from_fn(|i| h[i].max(n[i]))]
            });
        let center = kmul(0.5, add(l, h));
        for v in &mut self.v {
            *v = sub(*v, center);
        }
        let largest_val = self
            .v
            .iter()
            .fold(0., |m, v| v.into_iter().fold(m, |m, c| c.abs().max(m)));
        let scale = if largest_val == 0. {
            1.
        } else {
            largest_val.recip()
        };
        for v in &mut self.v {
            *v = kmul(scale, *v);
        }
        (scale, center)
    }
    /// Given a scale and translation output from normalization, reset the geometry to its
    /// original position.
    pub fn denormalize(&mut self, scale: F, trans: [F; 3]) {
        assert_ne!(scale, 0.);
        let inv_scale = scale.recip();
        for v in &mut self.v {
            *v = add(kmul(inv_scale, *v), trans);
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

        Self {
            v,
            face_mesh_idx: vec![0; fs.len()],
            f: fs,
            n,
            uv: [uv, vec![], vec![], vec![]],

            joint_idxs: vec![],
            joint_weights: vec![],
        }
    }
}

impl From<Obj> for Scene {
    fn from(obj: Obj) -> Self {
        let mut out = Self::default();
        for (name, mtl) in obj.mtls {
            let mut new_material = Material::default();
            new_material.name = name;
            if let Some(map_kd) = mtl.map_kd {
                new_material.textures.push((TextureKind::Diffuse, map_kd));
            }
            if let Some(map_ke) = mtl.map_ke {
                new_material.textures.push((TextureKind::Emissive, map_ke));
            }
            // TODO here add actual material details to material.
            out.materials.push(new_material);
        }
        for (i, o) in obj.objects.into_iter().enumerate() {
            out.meshes.push(Mesh::from(o));
            out.root_nodes.push(i);
            out.nodes.push(Node {
                mesh: Some(i),
                children: vec![],
                transform: super::identity::<4>(),
            });
        }
        out
    }
}

/// Convert a GLTF Scene into a flat mesh.
/// Will put the mesh into it's default pose.
#[cfg(feature = "gltf")]
impl From<super::gltf::GLTFScene> for Mesh {
    fn from(gltf_scene: super::gltf::GLTFScene) -> Self {
        let mut out = Self::default();
        gltf_scene.traverse(&mut |node, tform| {
            let Some(mi) = node.mesh else {
                return;
            };
            let mut mesh = gltf_scene.meshes[mi].clone();

            out.face_mesh_idx.extend((0..mesh.f.len()).map(|_| mi));
            let fs = mesh
                .f
                .iter()
                .map(|&vis| vis.map(|vi| vi + out.v.len()))
                .map(FaceKind::Tri);
            out.f.extend(fs);
            let curr_num_v = out.v.len();
            out.v
                .extend(mesh.v.into_iter().map(|v| super::tform_point(tform, v)));
            out.n.append(&mut mesh.n);
            out.uv[0].append(&mut mesh.uvs);
            if mesh.joint_idxs.is_empty() {
                assert!(mesh.joint_weights.is_empty());
                out.joint_idxs
                    .extend((curr_num_v..out.v.len()).map(|_| [0; 4]));
                out.joint_weights
                    .extend((curr_num_v..out.v.len()).map(|_| [0.; 4]));
            } else {
                assert!(!mesh.joint_weights.is_empty());
                out.joint_idxs.append(&mut mesh.joint_idxs);
                out.joint_weights.append(&mut mesh.joint_weights);
            }
        });
        // flip all UV
        for uv in &mut out.uv[0] {
            uv[1] = 1. - uv[1];
        }
        out
    }
}

#[cfg(feature = "gltf")]
impl From<super::gltf::GLTFMesh> for Mesh {
    fn from(gltf_mesh: super::gltf::GLTFMesh) -> Self {
        let super::gltf::GLTFMesh {
            v,
            f,
            uvs,
            n,
            joint_idxs,
            joint_weights,
        } = gltf_mesh;
        let uv = [uvs, vec![], vec![], vec![]];
        let f = f.into_iter().map(FaceKind::Tri).collect::<Vec<_>>();
        let face_mesh_idx = vec![];
        Self {
            v,
            f,
            uv,
            n,
            joint_idxs,
            joint_weights,
            face_mesh_idx,
        }
    }
}

#[cfg(feature = "gltf")]
impl From<super::gltf::GLTFScene> for Scene {
    fn from(gltf_scene: super::gltf::GLTFScene) -> Self {
        let mut out = Self::default();
        gltf_scene.traverse_with_parent(|| None, &mut |node, parent_idx: Option<usize>| {
            let mi = if let Some(mesh) = node.mesh {
                let mesh = gltf_scene.meshes[mesh].clone().into();
                let mi = out.meshes.len();
                out.meshes.push(mesh);
                Some(mi)
            } else {
                None
            };

            let new_node = Node {
                mesh: mi,
                children: vec![],
                transform: node.transform,
            };

            let own_idx = out.nodes.len();
            out.nodes.push(new_node);
            if let Some(parent_idx) = parent_idx {
                out.nodes[parent_idx].children.push(own_idx);
            } else {
                out.root_nodes.push(own_idx);
            }
            Some(own_idx)
        });
        out
    }
}
