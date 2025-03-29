use super::{Obj, ObjImage, ObjObject, MTL};
use crate::mesh::{Material, Mesh, Node, Scene, Texture, TextureKind, Transform};
use crate::{append_one, FaceKind};
use image::DynamicImage;

use std::array::from_fn;
use std::collections::HashMap;

impl From<ObjObject> for Mesh {
    fn from(obj: ObjObject) -> Self {
        if obj.vt.is_empty() && obj.vn.is_empty() {
            let v = obj.v;
            let f = obj
                .f
                .into_iter()
                .map(|pmf| FaceKind::from(pmf.v))
                .collect::<Vec<_>>();
            return Self {
                v,
                face_mesh_idx: vec![0; f.len()],
                f,
                face_mat_idx: obj.mat,
                ..Default::default()
            };
        }
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
                    (
                        f.v[$i],
                        f.vt.get($i).copied(),
                        // NOTE: Do not use normals to split edges, seems it's a bit buggy
                        //f.vn.get($i).copied().map(|vn| obj.vn[vn].map(F::to_bits)),
                    )
                };
            }
            for i in 0..f.v.len() {
                let key = key_i!(i);
                if !verts.contains_key(&key) {
                    v.push(obj.v[f.v[i]]);
                    if let Some(_vt) = key.1 {
                        uv.push(obj.vt[*f.vt.get(i).unwrap()]);
                    };
                    if let Some(&vn) = f.vn.get(i) {
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

            vert_colors: vec![],

            joint_idxs: vec![],
            joint_weights: vec![],
            name: String::new(),
            face_mat_idx: obj.mat,
        }
    }
}

impl From<Obj> for Scene {
    fn from(obj: Obj) -> Self {
        let mut out = Self::default();
        for (name, mtl) in obj.mtls {
            let mut mat = Material::from(mtl);
            mat.name = name;
            out.materials.push(mat);
        }
        for (i, o) in obj.objects.into_iter().enumerate() {
            out.meshes.push(Mesh::from(o));
            out.root_nodes.push(i);
            out.nodes.push(Node {
                mesh: Some(i),
                children: vec![],
                transform: Transform::ident_mat(),
                skin: None,
                name: String::new(),
                hidden: false,
            });
        }
        out.mtllibs = obj.mtllibs;
        out.input_file = obj.input_file;
        out
    }
}

fn to_parts(img: Option<ObjImage>) -> (Option<DynamicImage>, String) {
    let Some(img) = img else {
        return (None, String::new());
    };
    (Some(img.img), img.path)
}

impl From<MTL> for Material {
    fn from(mtl: MTL) -> Self {
        let mut mat = Material::default();
        let (kd, kd_path) = to_parts(mtl.map_kd);
        mat.textures.push(Texture {
            kind: TextureKind::Diffuse,
            mul: append_one(mtl.kd),
            image: kd,
            original_path: kd_path,
        });
        let (ks, ks_path) = to_parts(mtl.map_ks);
        mat.textures.push(Texture {
            kind: TextureKind::Specular,
            mul: append_one(mtl.ks),
            image: ks,
            original_path: ks_path,
        });
        let (ke, ke_path) = to_parts(mtl.map_ke);
        mat.textures.push(Texture {
            kind: TextureKind::Emissive,
            mul: append_one(mtl.ke),
            image: ke,
            original_path: ke_path,
        });
        let (normals, normals_path) = to_parts(mtl.bump_normal);
        mat.textures.push(Texture {
            kind: TextureKind::Normal,
            mul: [1.; 4],
            image: normals,
            original_path: normals_path,
        });
        mat
    }
}
