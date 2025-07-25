use super::{
    FBXCluster, FBXMaterial, FBXMesh, FBXMeshMaterial, FBXNode, FBXScene, FBXSettings, FBXSkin,
    VertexAttribute, id,
};
use crate::mesh::{
    Axis, Material, Mesh, Node, Scene, Settings, Skin, Texture, TextureKind, Transform,
};
use crate::{F, FaceKind, append_one};

use std::ops::Range;

use std::collections::HashMap;

fn uniq_vec_to_vertex_attribute<const N: usize>(
    v: &[[F; N]],
    num_vertices: usize,
) -> VertexAttribute<N> {
    if v.is_empty() {
        return Default::default();
    }
    use std::collections::{BTreeMap, btree_map::Entry};

    let mut uniq = BTreeMap::new();
    let mut uniq_vals = vec![];
    let mut idxs = vec![];
    let mut any_repeats = false;
    for &val in v {
        match uniq.entry(val.map(|v| v.to_bits())) {
            Entry::Vacant(v) => {
                v.insert(uniq_vals.len());
                idxs.push(uniq_vals.len());
                uniq_vals.push(val);
            }
            Entry::Occupied(o) => {
                any_repeats = true;
                idxs.push(*o.get())
            }
        }
    }

    if !any_repeats {
        assert_eq!(uniq_vals.len(), num_vertices, "{idxs:?}");
        // need to double check this is correct
        return VertexAttribute {
            values: uniq_vals,
            indices: vec![],

            ref_kind: super::RefKind::Direct,
            map_kind: super::VertexMappingKind::ByVertices,
        };
    }

    VertexAttribute {
        map_kind: if uniq_vals.len() < num_vertices {
            super::VertexMappingKind::ByVertices
        } else {
            super::VertexMappingKind::Wedge
        },
        values: uniq_vals,
        indices: idxs,

        ref_kind: super::RefKind::IndexToDirect,
    }
}

/*
/// Converts per vertex values to unique values and idxs into it.
fn to_idx_vecs<const N: usize>(vals: &[[F; N]]) -> (Vec<[F; N]>, Vec<usize>, bool) {
    use std::collections::{btree_map::Entry, BTreeMap};
    let mut uniq = BTreeMap::new();
    let mut uniq_vals = vec![];
    let mut idxs = vec![];
    let mut any_repeats = false;
    for &val in vals {
        match uniq.entry(val.map(|v| v.to_bits())) {
            Entry::Vacant(v) => {
                v.insert(uniq_vals.len());
                idxs.push(uniq_vals.len());
                uniq_vals.push(val);
            }
            Entry::Occupied(o) => {
                any_repeats = true;
                idxs.push(*o.get())
            }
        }
    }

    assert_eq!(idxs.len(), vals.len());
    (uniq_vals, idxs, any_repeats)
}
*/

fn condense_adjacent_values(v: &[usize]) -> Vec<(Range<usize>, usize)> {
    if v.is_empty() {
        return vec![];
    }

    let mut out = vec![];
    let mut curr_start = 0;
    let mut curr_end = 1;
    while curr_end < v.len() {
        if v[curr_end] == v[curr_end - 1] {
            curr_end += 1;
            continue;
        }
        out.push((curr_start..curr_end, v[curr_end - 1]));
        curr_start = curr_end;
        curr_end += 1;
    }
    out.push((curr_start..curr_end, *v.last().unwrap()));
    out
}

fn decompress_values(vs: Vec<(Range<usize>, usize)>) -> impl Iterator<Item = usize> {
    // assumes the ranges are sorted
    vs.into_iter().flat_map(|(r, v)| r.map(move |_| v))
}

#[test]
fn test_condense_adjacent_values() {
    assert_eq!(
        condense_adjacent_values(&[0, 0, 0, 0, 0, 0]),
        vec![(0..6, 0)]
    );
    assert_eq!(
        condense_adjacent_values(&[0, 0, 0, 0, 0, 1]),
        vec![(0..5, 0), (5..6, 1)]
    );
    assert_eq!(
        condense_adjacent_values(&[0, 0, 1, 0, 0, 1]),
        vec![(0..2, 0), (2..3, 1), (3..5, 0), (5..6, 1)]
    );
}

impl From<FBXSettings> for Settings {
    fn from(fbx_settings: FBXSettings) -> Settings {
        use Axis::*;
        let int_to_axis = |i, sign| match (i, sign) {
            (0, 1) => PosX,
            (1, 1) => PosY,
            (2, 1) => PosZ,
            (0, -1) => NegX,
            (1, -1) => NegY,
            (2, -1) => NegZ,
            _ => todo!("Unknown axis {i} or sign {sign}"),
        };
        Settings {
            up_axis: int_to_axis(fbx_settings.up_axis, fbx_settings.up_axis_sign),
            fwd_axis: int_to_axis(fbx_settings.front_axis, fbx_settings.front_axis_sign),
            tan_axis: int_to_axis(fbx_settings.coord_axis, fbx_settings.coord_axis_sign),
            scale: fbx_settings.unit_scale_factor as F,
        }
    }
}

impl From<Settings> for FBXSettings {
    fn from(settings: Settings) -> FBXSettings {
        use Axis::*;
        let axis_to_int = |axis| match axis {
            PosX | NegX => 0,
            PosY | NegY => 1,
            PosZ | NegZ => 2,
        };
        let axis_to_sign = |axis| match axis {
            PosX | PosY | PosZ => 1,
            NegX | NegY | NegZ => -1,
        };
        FBXSettings {
            up_axis: axis_to_int(settings.up_axis),
            up_axis_sign: axis_to_sign(settings.up_axis),

            front_axis: axis_to_int(settings.fwd_axis),
            front_axis_sign: axis_to_sign(settings.fwd_axis),

            coord_axis: axis_to_int(settings.tan_axis),
            coord_axis_sign: axis_to_sign(settings.tan_axis),

            og_up_axis: axis_to_int(settings.up_axis),
            og_up_axis_sign: axis_to_sign(settings.up_axis),

            unit_scale_factor: settings.scale as f64,
            og_unit_scale_factor: settings.scale as f64,

            time_span_start: 0,
            time_span_stop: 0,

            frame_rate: 60.,
        }
    }
}

/// Constructs a generic mesh from an FBXMesh, along with a vertex remapping.
impl From<FBXMesh> for (Mesh, Vec<usize>) {
    fn from(fbx_mesh: FBXMesh) -> Self {
        let FBXMesh {
            id: _,
            name,
            v,
            f,
            n,
            uv,
            mat,
            // unused
            color: _,
            skin: _,
        } = fbx_mesh;

        //let num_verts = f.iter().map(|f| f.len()).sum::<usize>();
        let num_faces = f.len();

        macro_rules! key_i {
            ($w: expr, $vi: expr) => {
                (
                    v[$vi].map(F::to_bits),
                    n.v($w, $vi).map(|n| n.map(F::to_bits)),
                    uv.v($w, $vi).map(|uv| uv.map(F::to_bits)),
                )
            };
        }

        let mut verts = HashMap::new();
        let mut remapping = vec![];

        let mut new_v = vec![];
        let mut new_uv = vec![];
        let mut new_n = vec![];
        let mut new_fs = vec![];

        let mut offset = 0;
        for f in f {
            let mut new_f = FaceKind::empty();
            for (o, &vi) in f.as_slice().iter().enumerate() {
                let key = key_i!(offset + o, vi);
                verts.entry(key).or_insert_with(|| {
                    new_v.push(v[vi]);
                    new_uv.extend(uv.v(offset + o, vi).into_iter());
                    new_n.extend(n.v(offset + o, vi).into_iter());

                    // store remapped vertex index
                    remapping.push(vi);
                    new_v.len() - 1
                });
                new_f.insert(verts[&key]);
            }
            new_fs.push(new_f);
            offset += f.len();
        }

        let mesh = Mesh {
            v: new_v,
            f: new_fs,
            n: new_n,
            uv: [new_uv, vec![], vec![], vec![]],
            name,
            face_mesh_idx: vec![],
            vert_colors: vec![],

            face_mat_idx: match mat {
                FBXMeshMaterial::None => vec![],
                FBXMeshMaterial::Global(i) => vec![(0..num_faces, i)],
                FBXMeshMaterial::PerFace(mats) => condense_adjacent_values(&mats),
            },

            joint_idxs: vec![],

            joint_weights: vec![],

            vertex_attrs: Default::default(),
        };
        (mesh, remapping)
    }
}

impl From<Mesh> for FBXMesh {
    fn from(mesh: Mesh) -> Self {
        let Mesh {
            v,
            f,
            n,
            uv,
            name,
            face_mesh_idx: _,

            face_mat_idx,

            joint_idxs: _,
            vert_colors: _,

            joint_weights: _,
            vertex_attrs: _,
        } = mesh;

        // FIXME? can change the output format if there are no repeats
        //let (n, vert_norm_idx, _any_n_repeats) = to_idx_vecs(&n);
        //let (uv, uv_idx, _) = to_idx_vecs(&uv[0]);

        let mat = if face_mat_idx.is_empty() {
            FBXMeshMaterial::None
        } else if face_mat_idx.len() == 1 {
            FBXMeshMaterial::Global(face_mat_idx[0].1)
        } else {
            FBXMeshMaterial::PerFace(decompress_values(face_mat_idx).collect::<Vec<_>>())
        };

        let num_verts = v.len();
        let n = uniq_vec_to_vertex_attribute(&n, num_verts);
        let uv = uniq_vec_to_vertex_attribute(&uv[0], num_verts);

        FBXMesh {
            id: id(),
            name,
            v,
            f,
            n,
            uv,

            mat,

            // unused
            color: Default::default(),
            skin: None,
        }
    }
}

impl From<FBXNode> for Node {
    fn from(fbx_node: FBXNode) -> Self {
        let FBXNode {
            id: _,
            mesh,
            children,
            name,
            transform,
            materials: _,
            cluster: _,
            parent: _,
            hidden,
            limb_node_id: _,
            is_null_node: _,
        } = fbx_node;

        Node {
            mesh,
            children,

            name,
            skin: None,
            transform: Transform::Decomposed(transform),
            hidden,
        }
    }
}

impl From<Node> for FBXNode {
    fn from(node: Node) -> Self {
        let Node {
            mesh,
            children,

            name,
            skin,
            transform,
            hidden,
        } = node;

        FBXNode {
            id: id(),
            mesh,
            children,
            parent: None,
            name,
            // materials must be computed externally
            materials: vec![],
            cluster: skin,
            transform: transform.to_decomposed(),
            hidden,
            limb_node_id: if skin.is_some() && mesh.is_none() {
                Some(id())
            } else {
                None
            },
            is_null_node: false,
        }
    }
}

impl From<(FBXSkin, &[FBXCluster])> for Skin {
    fn from((fbx_skin, fbx_clusters): (FBXSkin, &[FBXCluster])) -> Self {
        let FBXSkin {
            id: _,
            clusters: cluster_idxs,
            mesh: _,
            name: _,
            deform_acc: _,
        } = fbx_skin;

        let mut skin = Self::default();
        for &ci in &cluster_idxs {
            skin.joints.push(fbx_clusters[ci].node);
            // note sure if this is tform or tform link or tform link inverse?
            skin.inv_bind_matrices.push(fbx_clusters[ci].tform);

            // TODO need to figure out indices and weights, they are related to vertex weights.
        }

        skin
    }
}

impl FBXMaterial {
    pub fn to_textures(&self) -> [Option<Texture>; 1] {
        [(self.diffuse_color != [0.; 3]).then(|| {
            Texture::new(
                TextureKind::Diffuse,
                append_one(self.diffuse_color),
                None,
                String::new(),
            )
        })]
    }
}

/*
impl From<FBXMaterial> for Material {
    fn from(mat: FBXMaterial) -> Self {
        let mut textures = vec![];
        if mat.diffuse_color != [0.; 3] {
            let [dr, dg, db] = mat.diffuse_color;
            textures.push(Texture {
                kind: TextureKind::Diffuse,
                mul: [dr, dg, db, 1.],
                image: None,
                original_path: String::new(),
            })
        }

        // TODO specular here?

        Self {
            textures,
            name: mat.name.clone(),
            path: String::new(),
        }
    }
}
*/

impl From<FBXScene> for Scene {
    fn from(fbx_scene: FBXScene) -> Self {
        let mut out = Self::default();

        // construct empty joint weights and apply them later
        let jiws = fbx_scene
            .skins
            .iter()
            .map(|skin| {
                let mesh = &fbx_scene.meshes[skin.mesh];
                let mut joint_idxs = vec![[0; 4]; mesh.v.len()];
                let mut joint_ws = vec![[0.; 4]; mesh.v.len()];

                for &cl in &skin.clusters {
                    let cl = &fbx_scene.clusters[cl];
                    assert_eq!(cl.indices.len(), cl.weights.len());
                    for (&vi, &w) in cl.indices.iter().zip(cl.weights.iter()) {
                        let Some(slot) = joint_ws[vi].iter().position(|&v| v == 0.) else {
                            use std::sync::atomic::AtomicBool;
                            static DID_WARN: AtomicBool = AtomicBool::new(false);
                            if DID_WARN.fetch_or(true, std::sync::atomic::Ordering::SeqCst) {
                                continue;
                            }
                            eprintln!("[WARNING]: More than 4 influences for a single vertex ({vi}), will be culled");
                            continue;
                        };
                        //.expect("INTERNAL ERROR: More than 4 joint weights");
                        joint_idxs[vi][slot] = cl.node as u16;
                        joint_ws[vi][slot] = w;
                    }
                }

                (joint_idxs, joint_ws)
            })
            .collect::<Vec<_>>();

        let (meshes, remappings): (Vec<_>, Vec<_>) =
            fbx_scene.meshes.into_iter().map(Into::into).unzip();
        out.meshes = meshes;

        for (skin, (jis, jws)) in fbx_scene.skins.iter().zip(jiws.into_iter()) {
            let mesh = &mut out.meshes[skin.mesh];
            let remap = &remappings[skin.mesh];
            mesh.joint_idxs
                .extend(remap.iter().map(|&og_vi| jis[og_vi]));
            mesh.joint_weights
                .extend(remap.iter().map(|&og_vi| jws[og_vi]));
            // may also need to renormalize weights here

            assert_eq!(mesh.joint_idxs.len(), mesh.v.len());
            assert_eq!(mesh.joint_weights.len(), mesh.v.len());
        }

        // Materials for FBX meshes are stored with two levels of indirection
        // mesh has an index into node's material list, which indexes into scene's materials
        for node in &fbx_scene.nodes {
            let Some(mi) = node.mesh else {
                continue;
            };
            let mesh = &mut out.meshes[mi];

            for fm in &mut mesh.face_mat_idx {
                fm.1 = node.materials[fm.1];
            }

            // traverse up parent nodes, and get contributing index
        }
        out.nodes
            .extend(fbx_scene.nodes.into_iter().map(Into::into));

        out.skins.extend(
            fbx_scene
                .skins
                .into_iter()
                .map(|skin| (skin, fbx_scene.clusters.as_slice()))
                .map(Into::into),
        );

        for mat in &fbx_scene.materials {
            let textures = mat.to_textures();

            let textures = textures
                .into_iter()
                .flatten()
                .map(|txt| {
                    let ti = out.textures.len();
                    out.textures.push(txt);
                    ti
                })
                .collect::<Vec<_>>();

            let mat = Material {
                textures,
                name: mat.name.clone(),
                path: String::new(),
            };
            out.materials.push(mat);
        }

        out.root_nodes.extend_from_slice(&fbx_scene.root_nodes);
        out.settings = fbx_scene.global_settings.into();
        out
    }
}

impl From<Scene> for FBXScene {
    fn from(scene: Scene) -> Self {
        let mut out = Self::default();
        out.meshes.extend(scene.meshes.into_iter().map(Into::into));
        out.nodes.extend(scene.nodes.into_iter().map(Into::into));

        for ni in 0..out.nodes.len() {
            // need to fix parent nodes
            let children = std::mem::take(&mut out.nodes[ni].children);
            for &cn in &children {
                out.nodes[cn].parent = Some(ni);
            }
            let n = &mut out.nodes[ni];
            n.children = children;

            let Some(mi) = n.mesh else {
                continue;
            };

            let mesh = &mut out.meshes[mi];

            for &mati in mesh.mat.as_slice() {
                if !n.materials.contains(&mati) {
                    n.materials.push(mati)
                }
            }

            mesh.mat
                .remap(|v| n.materials.iter().position(|&p| p == v).unwrap());
        }

        out.root_nodes.extend_from_slice(&scene.root_nodes);
        out.global_settings = scene.settings.into();
        out
    }
}
