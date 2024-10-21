use super::obj::{Obj, ObjObject, MTL};
use super::{add, append_one, kmul, sub, FaceKind, F};

use std::array::from_fn;
use std::collections::HashMap;
use std::ops::Range;

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

#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub kind: TextureKind,
    pub mul: [F; 4],
    pub image: Option<image::DynamicImage>,
    pub original_path: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Material {
    pub textures: Vec<Texture>,
    pub name: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Skin {
    pub inv_bind_matrices: Vec<[[F; 4]; 4]>,
    pub joints: Vec<usize>,
    pub skeleton: Option<usize>,
    pub name: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Node {
    pub mesh: Option<usize>,
    pub children: Vec<usize>,
    pub transform: [[F; 4]; 4],
    pub skin: Option<usize>,
    pub name: String,
}

impl Node {
    pub fn traverse_with_parent<T>(
        &self,
        scene: &Scene,
        parent_val: T,
        visit: &mut impl FnMut(&Node, T) -> T,
    ) where
        T: Copy,
    {
        let new_val = visit(self, parent_val);
        for &c in &self.children {
            scene.nodes[c].traverse_with_parent(scene, new_val, visit);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Scene {
    pub root_nodes: Vec<usize>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub skins: Vec<Skin>,

    /// For an OBJ input, where are the MTL files
    pub(crate) mtllibs: Vec<String>,

    /// Path to the input file
    /// Needed for saving output later
    pub(crate) input_file: String,
}

impl Scene {
    pub fn traverse_with_parent<T>(
        &self,
        root_init: impl Fn() -> T,
        visit: &mut impl FnMut(&Node, T) -> T,
    ) where
        T: Copy,
    {
        for &root_node in &self.root_nodes {
            self.nodes[root_node].traverse_with_parent(self, root_init(), visit);
        }
    }
    /// Converts this scene into a flattened mesh which can then be repopulated back into a
    /// scene later.
    pub fn into_flattened_mesh(&self) -> Mesh {
        let mut out = Mesh::default();
        for (mi, m) in self.meshes.iter().enumerate() {
            let curr_vertex_offset = out.v.len();
            out.v.extend(m.v.iter().copied());
            for chan in 0..MAX_UV {
                out.uv[chan].extend(m.uv[chan].iter().copied());
            }
            out.n.extend(m.n.iter().copied());
            let curr_f = out.f.len();
            out.f.extend(m.f.iter().map(|f| {
                let mut f = f.clone();
                f.map(|vi| vi + curr_vertex_offset);
                f
            }));
            out.face_mesh_idx.extend(m.f.iter().map(|_| mi));
            out.face_mat_idx.extend(
                m.face_mat_idx
                    .iter()
                    .map(|(f, m)| ((f.start + curr_f)..(f.end + curr_f), *m)),
            );
            out.joint_idxs.extend(m.joint_idxs.iter().copied());
            out.joint_weights.extend(m.joint_weights.iter().copied());
        }
        out
    }
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

    /// Map of ranges for each face that correspond to a specific material
    pub face_mat_idx: Vec<(Range<usize>, usize)>,

    /// 1-1 relation between vertices and joint/idxs weights.
    pub joint_idxs: Vec<[u16; 4]>,
    pub joint_weights: Vec<[F; 4]>,

    /// Name of this mesh.
    pub name: String,
}

impl Mesh {
    pub fn flip_uv_v(&mut self) {
        for uv_chan in self.uv.iter_mut() {
            for uvs in uv_chan.iter_mut() {
                uvs[1] = 1. - uvs[1];
            }
        }
    }
    /// Returns the material for a given face if any.
    pub fn mat_for_face(&self, fi: usize) -> Option<usize> {
        self.face_mat_idx
            .iter()
            .find_map(|(fr, mi)| fr.contains(&fi).then_some(*mi))
    }
    pub fn num_tris(&self) -> usize {
        self.f.iter().map(|f| f.num_tris()).sum::<usize>()
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
            .fold(0., |m, v| v.iter().fold(m, |m, c| c.abs().max(m)));
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
    /// After flattening a scene into a mesh, repopulate the original scene with a modified
    /// flattened mesh.
    pub fn repopulate_scene(&self, scene: &mut Scene) {
        assert_eq!(
            self.face_mesh_idx.len(),
            self.f.len(),
            "Must specify which mesh belongs to which face"
        );

        // TODO maybe should also check that the number of meshes in the scene is greater than the
        // max number of meshes in the original mesh

        // mesh -> original_vertex_idx -> new_vertex_idx
        let mut vertex_map = vec![HashMap::new(); scene.meshes.len()];
        scene.meshes.fill_with(Default::default);
        // material for each mesh for each face
        let mut mat_map = vec![vec![]; scene.meshes.len()];

        for (fi, f) in self.f.iter().enumerate() {
            let mi = self.face_mesh_idx[fi];
            let mesh = &mut scene.meshes[mi];
            mat_map[mi].push(self.mat_for_face(fi));

            let mut f = f.clone();
            f.map(|flat_vi| {
                let new_vert_ins = || {
                    let vi = mesh.v.len();
                    mesh.v.push(self.v[flat_vi]);
                    for chan in 0..MAX_UV {
                        if let Some(&uv) = self.uv[chan].get(flat_vi) {
                            mesh.uv[chan].push(uv);
                            assert_eq!(mesh.v.len(), mesh.uv[chan].len());
                        }
                    }

                    if let Some(&n) = self.n.get(flat_vi) {
                        mesh.n.push(n);
                        assert_eq!(mesh.v.len(), mesh.n.len());
                    }

                    if let Some(&ji) = self.joint_idxs.get(flat_vi) {
                        mesh.joint_idxs.push(ji);
                        assert_eq!(mesh.v.len(), mesh.joint_idxs.len());
                    }

                    if let Some(&jw) = self.joint_weights.get(flat_vi) {
                        mesh.joint_weights.push(jw);
                        assert_eq!(mesh.v.len(), mesh.joint_weights.len());
                    }

                    vi
                };
                *vertex_map[mi].entry(flat_vi).or_insert_with(new_vert_ins)
            });

            mesh.f.push(f);
        }

        for (mi, mesh) in scene.meshes.iter_mut().enumerate() {
            mesh.face_mat_idx = convert_opt_usize(&mat_map[mi]);
        }
    }
}

// For converting optional material per face index to a range of faces.
pub fn convert_opt_usize(s: &[Option<usize>]) -> Vec<(Range<usize>, usize)> {
    let mut out = vec![];
    for (i, mati) in s.iter().enumerate() {
        let &Some(mati) = mati else {
            continue;
        };
        match out.last_mut() {
            None => out.push((i..(i + 1), mati)),
            Some((v, p_mati)) => {
                if v.end == i && mati == *p_mati {
                    v.end += 1;
                } else {
                    out.push((i..(i + 1), mati));
                }
            }
        }
    }

    out
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
                transform: super::identity::<4>(),
                skin: None,
                name: String::new(),
            });
        }
        out.mtllibs = obj.mtllibs;
        out.input_file = obj.input_file;
        out
    }
}

impl From<MTL> for Material {
    fn from(mtl: MTL) -> Self {
        let mut mat = Material::default();
        mat.textures.push(Texture {
            kind: TextureKind::Diffuse,
            mul: append_one(mtl.kd),
            image: mtl.map_kd,
            original_path: mtl.map_kd_path,
        });
        mat.textures.push(Texture {
            kind: TextureKind::Specular,
            mul: append_one(mtl.ks),
            image: mtl.map_ks,
            original_path: mtl.map_ks_path,
        });
        mat.textures.push(Texture {
            kind: TextureKind::Emissive,
            mul: append_one(mtl.ke),
            image: mtl.map_ke,
            original_path: mtl.map_ke_path,
        });
        mat.textures.push(Texture {
            kind: TextureKind::Normal,
            mul: [1.; 4],
            image: mtl.bump_normal,
            original_path: mtl.bump_normal_path,
        });
        mat
    }
}
