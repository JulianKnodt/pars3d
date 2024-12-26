use super::{id, FBXMesh, FBXMeshMaterial, FBXNode, FBXScene, FBXSettings};
use crate::mesh::{Axis, Mesh, Node, Scene, Settings, Transform};
use crate::{FaceKind, F};

use std::ops::Range;

use std::collections::HashMap;

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
        }
    }
}

impl From<FBXMesh> for Mesh {
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
        } = fbx_mesh;

        //let num_verts = f.iter().map(|f| f.len()).sum::<usize>();
        let num_faces = f.len();

        macro_rules! key_i {
            ($i: expr, $v: expr) => {
                (
                    $v.map(F::to_bits),
                    n.v($i).map(F::to_bits),
                    uv.v($i).map(F::to_bits),
                )
            };
        }

        let mut verts = HashMap::new();

        let mut new_v = vec![];
        let mut new_uv = vec![];
        let mut new_n = vec![];
        let mut new_fs = vec![];

        let mut offset = 0;
        for f in f {
            let mut new_f = FaceKind::empty();
            for (o, vi) in f.as_slice().iter().enumerate() {
                let key = key_i!(offset + o, v[*vi]);
                if !verts.contains_key(&key) {
                    new_v.push(v[*vi]);
                    new_uv.push(uv.v(offset + o));
                    new_n.push(n.v(offset + 0));
                    verts.insert(key, new_v.len() - 1);
                }
                new_f.insert(verts[&key]);
            }
            new_fs.push(new_f);
            offset += f.len();
        }

        let new_uv = [new_uv, vec![], vec![], vec![]];
        Mesh {
            v: new_v,
            f: new_fs,
            n: new_n,
            uv: new_uv,
            name,
            face_mesh_idx: vec![],

            face_mat_idx: match mat {
                FBXMeshMaterial::None => vec![],
                FBXMeshMaterial::Global(i) => vec![(0..num_faces, i)],
                FBXMeshMaterial::PerFace(mats) => condense_adjacent_values(&mats),
            },

            joint_idxs: vec![],

            joint_weights: vec![],
        }
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

            joint_weights: _,
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

        FBXMesh {
            id: id(),
            name,
            v,
            f,
            n: Default::default(),
            uv: Default::default(),

            mat,
            // unused
            color: Default::default(),
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
            skin: _,
        } = fbx_node;
        Node {
            mesh,
            children,

            name,
            skin: None,
            transform: Transform::Decomposed(transform),
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
        } = node;

        FBXNode {
            id: id(),
            mesh,
            children,
            name,
            // materials must be computed externally
            materials: vec![],
            skin,
            transform: transform.to_decomposed(),
        }
    }
}

impl From<FBXScene> for Scene {
    fn from(fbx_scene: FBXScene) -> Self {
        let mut out = Self::default();
        out.meshes
            .extend(fbx_scene.meshes.into_iter().map(Into::into));
        // Materials for FBX meshes are stored with two levels of indirection
        // mesh has an index into node's material list, which indexes into scene's materials
        for node in &fbx_scene.nodes {
            let Some(mi) = node.mesh else {
                continue;
            };
            for fm in &mut out.meshes[mi].face_mat_idx {
                fm.1 = node.materials[fm.1];
            }
        }
        out.nodes
            .extend(fbx_scene.nodes.into_iter().map(Into::into));

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

        for n in &mut out.nodes {
            let Some(mi) = n.mesh else {
                continue;
            };

            for &mati in out.meshes[mi].mat.as_slice() {
                if !n.materials.contains(&mati) {
                    n.materials.push(mati)
                }
            }

            out.meshes[mi]
                .mat
                .remap(|v| n.materials.iter().position(|&p| p == v).unwrap());
        }

        out.root_nodes.extend_from_slice(&scene.root_nodes);
        out.global_settings = scene.settings.into();
        out
    }
}
