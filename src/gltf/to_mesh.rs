use super::gltf::{GLTFMesh, GLTFNode, GLTFScene, GLTFSkin};
use crate::mesh::{Mesh, Node, Scene, Skin, Transform};

use crate::FaceKind;

impl From<GLTFMesh> for Mesh {
    fn from(gltf_mesh: GLTFMesh) -> Self {
        let GLTFMesh {
            v,
            f,
            uvs,
            n,
            face_mat_idx,
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
            face_mat_idx,

            name: String::new(),
            vert_colors: vec![],
            vertex_attrs: Default::default(),
            l: vec![],
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
        out.animations = gltf_scene.animations;
        out.meshes
            .extend(gltf_scene.meshes.into_iter().map(|mesh| mesh.into()));
        // for each mesh, also ensure that if any has joint then all will have joints
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
                transform: Transform::Matrix(transform),
                skin,
                name,
                hidden: false,
            }
        }));
        out.root_nodes = gltf_scene.root_nodes.clone();
        out
    }
}
