use super::{FBXMesh, FBXNode, FBXScene};
use crate::mesh::{Mesh, Node, Scene};
use crate::F;
use std::sync::atomic::AtomicUsize;

use std::collections::{btree_map::Entry, BTreeMap};

fn id() -> usize {
    static mut CURR_ID: AtomicUsize = AtomicUsize::new(3333);
    let id = unsafe {
        #[allow(static_mut_refs)]
        CURR_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    };
    assert_ne!(id, 0);
    id
}

/// Converts per vertex values to unique values and idxs into it.
fn to_idx_vecs<const N: usize>(vals: &[[F; N]]) -> (Vec<[F; N]>, Vec<usize>) {
    let mut uniq = BTreeMap::new();
    let mut uniq_vals = vec![];
    let mut idxs = vec![];
    for &val in vals {
        match uniq.entry(val.map(|v| v.to_bits())) {
            Entry::Vacant(v) => {
                v.insert(uniq_vals.len());
                idxs.push(uniq_vals.len());
                uniq_vals.push(val);
            }
            Entry::Occupied(o) => idxs.push(*o.get()),
        }
    }

    assert_eq!(idxs.len(), vals.len());
    (uniq_vals, idxs)
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

        let n = vert_norm_idx
            .into_iter()
            .map(|ni| n[ni])
            .collect::<Vec<_>>();
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

        let (n, vert_norm_idx) = to_idx_vecs(&n);
        let (uv, uv_idx) = to_idx_vecs(&uv[0]);

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
        } = fbx_node;
        Node {
            mesh,
            children,

            name,
            skin: None,
            transform: crate::identity(),
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
            transform: _,
        } = node;

        FBXNode {
            id: id(),
            mesh,
            children,
            name,
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
        out
    }
}

impl From<Scene> for FBXScene {
    fn from(scene: Scene) -> Self {
        let mut out = Self::default();
        out.meshes.extend(scene.meshes.into_iter().map(Into::into));
        out.nodes.extend(scene.nodes.into_iter().map(Into::into));
        out.root_nodes.extend_from_slice(&scene.root_nodes);
        out
    }
}
