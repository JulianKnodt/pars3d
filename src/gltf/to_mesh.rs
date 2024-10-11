use super::gltf::{GLTFMesh, GLTFScene};
use crate::mesh::{Mesh, Node, Scene};

use crate::{tform_point, FaceKind};

/// Convert a GLTF Scene into a flat mesh.
/// Will put the mesh into it's default pose.
impl From<GLTFScene> for Mesh {
    fn from(gltf_scene: GLTFScene) -> Self {
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
                .extend(mesh.v.into_iter().map(|v| tform_point(tform, v)));
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

impl From<GLTFMesh> for Mesh {
    fn from(gltf_mesh: GLTFMesh) -> Self {
        let GLTFMesh {
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

impl From<GLTFScene> for Scene {
    fn from(gltf_scene: GLTFScene) -> Self {
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
                // TODO implement skins here
                skin: None,
                name: node.name.clone(),
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
