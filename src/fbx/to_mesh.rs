use super::{FBXMesh, FBXNode, FBXScene};
use crate::mesh::{Mesh, Node, Scene};

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

impl From<FBXScene> for Scene {
    fn from(fbx_scene: FBXScene) -> Self {
        let mut out = Self::default();
        out.meshes
            .extend(fbx_scene.meshes.into_iter().map(Into::into));
        out.nodes.extend(fbx_scene.nodes.into_iter().map(|node| {
            let FBXNode {
                id: _,
                mesh,
                children,
                name,
            } = node;
            Node {
                mesh,
                children,

                name,
                skin: None,
                transform: crate::identity(),
            }
        }));
        out.root_nodes.extend_from_slice(&fbx_scene.root_nodes);
        out
    }
}
