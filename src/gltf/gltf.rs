use crate::{identity, matmul, F};
use gltf_json::validation::{Checked::Valid, USize64};
use std::io::{self, Write};
use std::mem;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFScene {
    pub nodes: Vec<GLTFNode>,
    pub meshes: Vec<GLTFMesh>,

    pub root_nodes: Vec<usize>,
    pub skins: Vec<GLTFSkin>, //materials: Vec<Material>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFSkin {
    pub name: String,

    pub inv_bind_matrices: Vec<[[F; 4]; 4]>,

    pub joints: Vec<usize>,

    pub skeleton: Option<usize>,
}

impl GLTFScene {
    pub fn traverse(&self, visit: &mut impl FnMut(&GLTFNode, [[F; 4]; 4])) {
        for &root_node in &self.root_nodes {
            self.nodes[root_node].traverse(self, identity::<4>(), visit);
        }
    }
    pub fn traverse_with_parent<T>(
        &self,
        root_init: impl Fn() -> T,
        visit: &mut impl FnMut(&GLTFNode, T) -> T,
    ) where
        T: Copy,
    {
        for &root_node in &self.root_nodes {
            self.nodes[root_node].traverse_with_parent(self, root_init(), visit);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFNode {
    pub mesh: Option<usize>,
    pub children: Vec<usize>,
    pub skin: Option<usize>,
    /// index from source do not change
    pub(crate) index: usize,

    pub name: String,
    // TODO also needs to include a transform
    pub transform: [[F; 4]; 4],
}

impl GLTFNode {
    pub fn traverse(
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
    pub fn traverse_with_parent<T>(
        &self,
        scene: &GLTFScene,
        parent_val: T,
        visit: &mut impl FnMut(&GLTFNode, T) -> T,
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
        new_node.index = node.index();
        new_node.name = node.name().unwrap_or("").into();
        new_node.transform = identity::<4>();

        if let Some(s) = node.skin() {
            new_node.skin = Some(out.skins.len());
            let mut new_skin = GLTFSkin::default();
            let bind_mats = s
                .reader(|buffer: gltf::Buffer| buffers.get(buffer.index()).map(|data| &data[..]))
                .read_inverse_bind_matrices()
                .into_iter()
                .flatten()
                .map(|p| p.map(|col| col.map(|v| v as F)));
            new_skin.inv_bind_matrices.extend(bind_mats);
            new_skin.name = s.name().unwrap_or("").into();
            new_skin.joints.extend(s.joints().map(|j| j.index()));
            out.skins.push(new_skin);
        }

        if let Some(m) = node.mesh() {
            let mut new_mesh = GLTFMesh::default();
            new_node.transform = node.transform().matrix().map(|col| col.map(|v| v as F));
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
                if !new_mesh.uvs.is_empty() {
                    assert_eq!(new_mesh.uvs.len(), new_mesh.v.len());
                }

                let ns = reader
                    .read_normals()
                    .into_iter()
                    .flatten()
                    .map(|n| n.map(|n| n as F));
                new_mesh.n.extend(ns);
                if !new_mesh.n.is_empty() {
                    assert_eq!(new_mesh.n.len(), new_mesh.v.len());
                }

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
            // correct original index to new index.
            // TODO may want to check names are the same as well? But it's less reliable.
            for s in out.skins.iter_mut() {
                for j in s.joints.iter_mut() {
                    *j = out
                        .nodes
                        .iter()
                        .position(|n| n.index == *j)
                        .expect("Could not find matching node");
                }
            }
            out
        })
        .collect::<Vec<_>>();
    Ok(out)
}

/// Save a scene as a glb file (binary GLTF file).
pub fn save_glb(scene: &crate::mesh::Scene, dst: impl Write) -> io::Result<()> {
    use std::borrow::Cow;
    let mut root = gltf_json::Root::default();

    let mut bytes: Vec<u8> = vec![];

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    struct Vertex {
        // note that these are specified by the GLTF spec not by pars3d.
        v: [f32; 3],
        uv: [f32; 2],
        n: [f32; 3],
        joint_weights: [f32; 4],
        joint_idxs: [u16; 4],
    }
    impl Vertex {
        pub fn to_bytes(&self) -> impl Iterator<Item = u8> + '_ {
            self.v
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .chain(self.uv.into_iter().flat_map(f32::to_le_bytes))
                .chain(self.n.into_iter().flat_map(f32::to_le_bytes))
                .chain(self.joint_weights.into_iter().flat_map(f32::to_le_bytes))
                .chain(self.joint_idxs.into_iter().flat_map(u16::to_le_bytes))
        }
    }
    // TODO make 2 buffers per mesh instead of one giant buffer of everything.
    let mut verts = vec![];
    for mesh in &scene.meshes {
        for i in 0..mesh.v.len() {
            let v = Vertex {
                v: mesh.v[i].map(|v| v as f32),
                uv: mesh.uv[0]
                    .get(i)
                    .copied()
                    .unwrap_or_default()
                    .map(|v| v as f32),
                n: mesh.n.get(i).copied().unwrap_or_default().map(|v| v as f32),
                joint_idxs: mesh.joint_idxs.get(i).copied().unwrap_or_default(),
                joint_weights: mesh
                    .joint_weights
                    .get(i)
                    .copied()
                    .unwrap_or_default()
                    .map(|v| v as f32),
            };
            bytes.extend(v.to_bytes());
            verts.push(v);
        }
    }
    let f_offset = bytes.len();
    let mut n_tris = 0usize;
    let mut vertex_offset = 0;
    for mesh in &scene.meshes {
        for f in &mesh.f {
            for tri in f.as_triangle_fan() {
                let raw = tri
                    .into_iter()
                    .flat_map(|vi| (vi as u32 + vertex_offset).to_le_bytes());
                bytes.extend(raw);
                n_tris += 1;
            }
        }
        vertex_offset += mesh.v.len() as u32;
    }
    let [lb, ub] = verts
        .iter()
        .fold([[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]], |[l, h], n| {
            [
                std::array::from_fn(|i| l[i].min(n.v[i])),
                std::array::from_fn(|i| h[i].max(n.v[i])),
            ]
        });

    let buf_len = bytes.len();
    let buffer = root.push(gltf_json::Buffer {
        byte_length: USize64::from(buf_len),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: None, //ascii.then(|| String::from("buffer0.bin")),
    });

    let buffer_view = root.push(gltf_json::buffer::View {
        buffer,
        byte_length: USize64::from(f_offset),
        byte_offset: None,
        byte_stride: Some(gltf_json::buffer::Stride(mem::size_of::<Vertex>())),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(gltf_json::buffer::Target::ArrayBuffer)),
    });
    let idx_buffer_view = root.push(gltf_json::buffer::View {
        buffer,
        byte_length: USize64::from(buf_len - f_offset),
        byte_offset: Some(USize64::from(f_offset)),
        byte_stride: None,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(gltf_json::buffer::Target::ElementArrayBuffer)),
    });
    let positions = root.push(gltf_json::Accessor {
        buffer_view: Some(buffer_view),
        byte_offset: Some(USize64(0)),
        count: USize64::from(verts.len()),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Vec3),
        min: Some(gltf_json::Value::from(Vec::from(lb))),
        max: Some(gltf_json::Value::from(Vec::from(ub))),
        name: None,
        normalized: false,
        sparse: None,
    });
    let uvs = root.push(gltf_json::Accessor {
        buffer_view: Some(buffer_view),
        byte_offset: Some(USize64::from(3 * mem::size_of::<f32>())),
        count: USize64::from(verts.len()),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Vec2),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    let normals = root.push(gltf_json::Accessor {
        buffer_view: Some(buffer_view),
        byte_offset: Some(USize64::from(5 * mem::size_of::<f32>())),
        count: USize64::from(verts.len()),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Vec3),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    let joint_ws = root.push(gltf_json::Accessor {
        buffer_view: Some(buffer_view),
        byte_offset: Some(USize64::from(8 * mem::size_of::<f32>())),
        count: USize64::from(verts.len()),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::F32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Vec4),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    let joint_idxs = root.push(gltf_json::Accessor {
        buffer_view: Some(buffer_view),
        byte_offset: Some(USize64::from(12 * mem::size_of::<f32>())),
        count: USize64::from(verts.len()),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::U16,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Vec4),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    let faces = root.push(gltf_json::Accessor {
        buffer_view: Some(idx_buffer_view),
        byte_offset: Some(USize64(0)),
        count: USize64::from(n_tris * 3),
        component_type: Valid(gltf_json::accessor::GenericComponentType(
            gltf_json::accessor::ComponentType::U32,
        )),
        extensions: Default::default(),
        extras: Default::default(),
        type_: Valid(gltf_json::accessor::Type::Scalar),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });
    let primitive = gltf_json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(Valid(gltf_json::mesh::Semantic::Positions), positions);
            map.insert(Valid(gltf_json::mesh::Semantic::TexCoords(0)), uvs);
            map.insert(Valid(gltf_json::mesh::Semantic::Normals), normals);
            map.insert(Valid(gltf_json::mesh::Semantic::Joints(0)), joint_idxs);
            map.insert(Valid(gltf_json::mesh::Semantic::Weights(0)), joint_ws);
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: Some(faces),
        material: None,
        mode: Valid(gltf_json::mesh::Mode::Triangles),
        targets: None,
    };

    let mesh = root.push(gltf_json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        primitives: vec![primitive],
        weights: None,
    });

    let mut nodes = vec![];
    for (ni, n) in scene.nodes.iter().enumerate() {
        let matrix = if n.transform == identity::<4>() {
            None
        } else {
            Some(unsafe { std::mem::transmute(n.transform.map(|col| col.map(|v| v as f32))) })
        };
        let children = if n.children.is_empty() {
            None
        } else {
            let children = n
                .children
                .iter()
                .map(|&v| gltf_json::Index::new(v as u32))
                .collect::<Vec<_>>();
            Some(children)
        };
        let node = root.push(gltf_json::Node {
            // TODO actually write out each mesh as separate.
            mesh: (ni == 0).then_some(mesh),
            name: (!n.name.is_empty()).then(|| n.name.clone()),

            skin: None,
            children,
            matrix,
            ..Default::default()
        });

        nodes.push(node);
    }

    root.push(gltf_json::Scene {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        nodes: scene
            .root_nodes
            .iter()
            .map(|&v| gltf_json::Index::new(v as u32))
            .collect::<Vec<_>>(),
    });

    let json_string = gltf_json::serialize::to_string(&root).expect("Serialization error");
    let mut json_offset = json_string.len();
    align_to_multiple_of_four(&mut json_offset);
    let glb = gltf::binary::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            // N.B., the size of binary glTF file is limited to range of `u32`.
            length: (json_offset + buf_len)
                .try_into()
                .expect("file size exceeds binary glTF limit"),
        },
        bin: Some(Cow::Owned(to_padded_byte_vector(bytes))),
        json: Cow::Owned(json_string.into_bytes()),
    };
    glb.to_writer(dst).expect("glTF binary output error");

    Ok(())
}

fn align_to_multiple_of_four(n: &mut usize) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * mem::size_of::<T>();
    let byte_capacity = vec.capacity() * mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
}
#[test]
fn test_load_gltf() {
    let mut scenes = load("etrian_odyssey_3_monk.glb").unwrap();
    assert_eq!(scenes.len(), 1);
    let scene = scenes.pop().unwrap();
    assert_eq!(scene.root_nodes.len(), 1);
    assert_eq!(scene.meshes.len(), 24);
    assert_eq!(scene.nodes.len(), 293);
    assert_eq!(scene.skins.len(), 18);
}

#[test]
fn test_gltf_load_save() {
    let mut scenes = load("etrian_odyssey_3_monk.glb").expect("Failed to open");
    assert_eq!(scenes.len(), 1);
    let scene = scenes.pop().unwrap();
    let out = std::fs::File::create("gltf_test.glb").expect("Failed to create file");
    save_glb(&crate::mesh::Scene::from(scene), io::BufWriter::new(out))
        .expect("Failed to write file");
}
