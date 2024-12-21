use super::{id, FBXMesh, FBXNode, FBXScene, FBXSettings};
use crate::mesh::{Axis, Mesh, Node, Scene, Settings, Transform};
use crate::F;

use std::collections::{btree_map::Entry, BTreeMap};

/// Converts per vertex values to unique values and idxs into it.
fn to_idx_vecs<const N: usize>(vals: &[[F; N]]) -> (Vec<[F; N]>, Vec<usize>, bool) {
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
            vert_norm_idx,
            uv,
            uv_idx,
            global_mat: _,
            per_face_mat: _,
            // unused
            vertex_colors: _,
            vertex_color_idx: _,
        } = fbx_mesh;

        let n = if vert_norm_idx.is_empty() {
            n
        } else {
            vert_norm_idx
                .into_iter()
                .map(|ni| n[ni])
                .collect::<Vec<_>>()
        };
        let uv0 = uv_idx.into_iter().map(|uvi| uv[uvi]).collect::<Vec<_>>();
        let uv = [uv0, vec![], vec![], vec![]];
        Mesh {
            v,
            f,
            n,
            uv,
            name,
            face_mesh_idx: vec![],

            // TODO fill this in
            face_mat_idx: vec![],

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

            // TODO handle this
            face_mat_idx: _,

            joint_idxs: _,

            joint_weights: _,
        } = mesh;

        let (n, vert_norm_idx, _any_n_repeats) = to_idx_vecs(&n);
        // FIXME? can change the output format if there are no repeats
        let (uv, uv_idx, _) = to_idx_vecs(&uv[0]);

        FBXMesh {
            id: id(),
            name,
            v,
            f,
            n,
            vert_norm_idx,

            uv,
            uv_idx,

            global_mat: None,
            per_face_mat: vec![],
            // unused
            vertex_colors: vec![],
            vertex_color_idx: vec![],
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
            skin: _,
            transform,
        } = node;

        FBXNode {
            id: id(),
            mesh,
            children,
            name,
            transform: transform.to_decomposed(),
        }
    }
}

impl From<FBXScene> for Scene {
    fn from(fbx_scene: FBXScene) -> Self {
        let mut out = Self::default();
        out.meshes
            .extend(fbx_scene.meshes.into_iter().map(Into::into));
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
        out.root_nodes.extend_from_slice(&scene.root_nodes);
        out.global_settings = scene.settings.into();
        out
    }
}
