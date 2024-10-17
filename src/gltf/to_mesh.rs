use super::gltf::{GLTFMesh, GLTFNode, GLTFScene, GLTFSkin};
use crate::mesh::{Mesh, Node, Scene, Skin};

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
            name: String::new(),
        }
    }
}

impl From<GLTFScene> for Scene {
    fn from(gltf_scene: GLTFScene) -> Self {
        let mut out = Self::default();
        out.skins.extend(gltf_scene.skins.into_iter().map(|skin| {
            let GLTFSkin {
                inv_bind_matrices,
                name,
                joints,
                skeleton,
            } = skin;
            Skin {
                inv_bind_matrices,
                name,
                joints,
                skeleton,
            }
        }));
        out.meshes
            .extend(gltf_scene.meshes.into_iter().map(|mesh| mesh.into()));
        out.nodes.extend(gltf_scene.nodes.into_iter().map(|node| {
            let GLTFNode {
                mesh,
                children,
                skin,
                name,
                transform,
                ..
            } = node;
            Node {
                mesh,
                children,
                transform,
                skin,
                name,
            }
        }));
        out.root_nodes = gltf_scene.root_nodes.clone();
        out
    }
}
