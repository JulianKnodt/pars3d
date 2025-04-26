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

        let must_match_uvs = obj.f.iter().any(|f| !f.vt.is_empty());
        let must_match_nrm = obj.f.iter().any(|f| !f.vn.is_empty());

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
                    if let Some(vt) = key.1 {
                        uv.push(obj.vt[vt]);
                    } else if must_match_uvs {
                        uv.push([0.; 2]);
                    }
                    if let Some(&vn) = f.vn.get(i) {
                        n.push(obj.vn[vn]);
                    } else if must_match_nrm {
                        n.push([0.; 3]);
                    }
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
            let textures = mtl.to_textures().map(|txt| {
                let ti = out.textures.len();
                out.textures.push(txt);
                ti
            });
            let mat = Material {
                textures: textures.to_vec(),
                name,
                ..Default::default()
            };
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

impl MTL {
    pub fn to_textures(self) -> [Texture; 4] {
        let texture = |kind, mul, image, original_path| Texture {
            kind,
            mul,
            image,
            original_path,
        };
        let (kd, kd_path) = to_parts(self.map_kd);
        let (ks, ks_path) = to_parts(self.map_ks);
        let (ke, ke_path) = to_parts(self.map_ke);
        let (normals, normals_path) = to_parts(self.bump_normal);
        [
            texture(TextureKind::Diffuse, append_one(self.kd), kd, kd_path),
            texture(TextureKind::Specular, append_one(self.ks), ks, ks_path),
            texture(TextureKind::Emissive, append_one(self.ke), ke, ke_path),
            texture(TextureKind::Normal, [1.; 4], normals, normals_path),
        ]
    }
}

/*
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
*/
