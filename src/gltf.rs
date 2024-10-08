use super::{identity, matmul, F};
use std::io;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFScene {
    pub nodes: Vec<GLTFNode>,
    pub meshes: Vec<GLTFMesh>,

    pub root_nodes: Vec<usize>,
    //materials: Vec<Material>,
}

impl GLTFScene {
    pub fn traverse(&self, visit: &mut impl FnMut(&GLTFNode, [[F; 4]; 4])) {
        for &root_node in &self.root_nodes {
            self.nodes[root_node].traverse(self, identity::<4>(), visit);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFNode {
    pub mesh: Option<usize>,
    children: Vec<usize>,
    skin: Vec<usize>,

    name: String,
    // TODO also needs to include a transform
    pub transform: [[F; 4]; 4],
}

impl GLTFNode {
    fn traverse(
        &self,
        scene: &GLTFScene,
        curr_tform: [[F; 4]; 4],
        visit: &mut impl FnMut(&Self, [[F; 4]; 4]),
    ) {
        let new_tform = matmul(self.transform, curr_tform);
        visit(self, new_tform);
        for &c in &self.children {
            scene.nodes[c].traverse(scene, new_tform, visit);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFMesh {
    pub v: Vec<[F; 3]>,
    pub f: Vec<[usize; 3]>,

    pub uvs: Vec<[F; 2]>,

    pub n: Vec<[F; 3]>,

    // For each vertex, associate it with 4 bones
    pub joint_idxs: Vec<[u16; 4]>,
    pub joint_weights: Vec<[F; 4]>,
}

/// Load a GLTF/GLB file into a GLTFScene.
pub fn load<P>(path: P) -> gltf::Result<Vec<GLTFScene>>
where
    P: AsRef<std::path::Path>,
{
    let (doc, buffers, _images) = gltf::import(path)?;

    fn traverse_node(
        gltf: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        node: &gltf::scene::Node,
        out: &mut GLTFScene,
    ) -> usize {
        let mut new_node = GLTFNode::default();
        new_node.name = node.name().map(String::from).unwrap_or_else(String::new);
        new_node.transform = identity::<4>();

        if let Some(m) = node.mesh() {
            let mut new_mesh = GLTFMesh::default();
            new_node.transform = node.transform().matrix().map(|col| col.map(|v| v as F));
            println!("{:?}", new_node.transform);
            for p in m.primitives() {
                let offset = new_mesh.v.len();
                let reader = p.reader(|buffer: gltf::Buffer| {
                    buffers.get(buffer.index()).map(|data| &data[..])
                });
                let ps = reader
                    .read_positions()
                    .into_iter()
                    .flatten()
                    .map(|p| p.map(|v| v as F));
                new_mesh.v.extend(ps);

                let uvs = reader
                    .read_tex_coords(0)
                    .map(|v| v.into_f32())
                    .into_iter()
                    .flatten()
                    .map(|uvs| uvs.map(|uv| uv as F));
                new_mesh.uvs.extend(uvs);
                assert_eq!(new_mesh.uvs.len(), new_mesh.v.len());

                let ns = reader
                    .read_normals()
                    .into_iter()
                    .flatten()
                    .map(|n| n.map(|n| n as F));
                new_mesh.n.extend(ns);
                assert_eq!(new_mesh.n.len(), new_mesh.v.len());

                if let Some(jr) = reader.read_joints(0) {
                    new_mesh.joint_idxs.extend(jr.into_u16());
                    assert_eq!(new_mesh.joint_idxs.len(), new_mesh.v.len());
                    let jwr = reader
                        .read_weights(0)
                        .expect("GLTF has joints but no weights?");
                    new_mesh
                        .joint_weights
                        .extend(jwr.into_f32().map(|ws| ws.map(|w| w as F)));
                }

                let idxs = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .array_chunks::<3>();
                new_mesh
                    .f
                    .extend(idxs.map(|vis| vis.map(|vi| vi as usize + offset)));
            }
            new_node.mesh = Some(out.meshes.len());
            out.meshes.push(new_mesh);
        }
        for child in node.children() {
            let child_idx = traverse_node(gltf, buffers, &child, out);
            new_node.children.push(child_idx);
        }
        let idx = out.nodes.len();
        out.nodes.push(new_node);
        idx
    }

    let out = doc
        .scenes()
        .map(|scene| {
            let mut out = GLTFScene::default();
            for root_node in scene.nodes() {
                let idx = traverse_node(&doc, &buffers, &root_node, &mut out);
                out.root_nodes.push(idx);
            }
            out
        })
        .collect::<Vec<_>>();
    Ok(out)
}

// TODO Seems this is non-trivial for more complicated meshes.
#[allow(unused)]
fn save_binary(_scene: &GLTFScene, _r: impl std::io::Read, _binary: bool) -> io::Result<()> {
    todo!()
    //let mut root = gltf_json::Root::default();
    //let buffer_len = scene.meshes.iter().map(|v|
}

#[test]
fn test_load_gltf() {
    let mut scenes = load("etrian_odyssey_3_monk.glb").unwrap();
    assert_eq!(scenes.len(), 1);
    let scene = scenes.pop().unwrap();
    assert_eq!(scene.root_nodes.len(), 1);
    assert_eq!(scene.meshes.len(), 24);
    assert_eq!(scene.nodes.len(), 293);
    /*
    for [x, y, z] in &mesh.v {
        println!("v {x} {y} {z}");
    }
    for ijk in &mesh.f {
        let [i, j, k] = ijk.map(|i| i + 1);
        println!("f {i} {j} {k}");
    }
    */
}
