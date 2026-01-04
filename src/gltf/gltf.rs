use crate::anim::{
    Animation, Channel, Dim, InterpolationKind, OutputProperty, Sampler, Samplers, Time,
};
use crate::{F, identity, matmul};
use gltf_json::validation::{Checked::Valid, USize64};
use std::io::{self, Write};
use std::mem;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFScene {
    pub nodes: Vec<GLTFNode>,
    pub meshes: Vec<GLTFMesh>,

    pub materials: Vec<GLTFMaterial>,

    pub root_nodes: Vec<usize>,
    pub skins: Vec<GLTFSkin>, //materials: Vec<Material>,

    pub animations: Vec<Animation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageData {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    format: gltf::image::Format,
}

fn to_f32(v: F) -> f32 {
    v as f32
}

impl From<gltf::image::Data> for ImageData {
    fn from(d: gltf::image::Data) -> ImageData {
        let gltf::image::Data {
            pixels,
            width,
            height,
            format,
        } = d;
        ImageData {
            pixels,
            width,
            height,
            format,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TexInfo {
    uv_channel: u32,
    // TODO this can either be a URI w/ optional mime OR raw buffer w/ uri
    texture: ImageData,
    mime_type: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PbrMetallicRoughness {
    base_color_factor: [F; 4],
    base_color_texture: Option<TexInfo>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct GLTFMaterial {
    pbr_metallic_roughness: PbrMetallicRoughness,
    name: String,
    double_sided: bool,
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

    pub face_mat_idx: Vec<(std::ops::Range<usize>, usize)>,

    // For each vertex, associate it with 4 bones
    pub joint_idxs: Vec<[u16; 4]>,
    pub joint_weights: Vec<[F; 4]>,
}

/// Load a GLTF/GLB file into a GLTFScene.
pub fn load<P>(path: P) -> gltf::Result<GLTFScene>
where
    P: AsRef<std::path::Path>,
{
    let (doc, buffers, images) = gltf::import(path)?;

    let mut out = GLTFScene::default();
    for scene in doc.scenes() {
        for root_node in scene.nodes() {
            out.root_nodes.push(root_node.index());
        }
    }

    for mat in doc.materials() {
        let mut new_mat = GLTFMaterial {
            name: mat.name().map(String::from).unwrap_or_else(String::new),
            ..Default::default()
        };
        let prev_pbr_mr = mat.pbr_metallic_roughness();
        new_mat.pbr_metallic_roughness = PbrMetallicRoughness {
            base_color_factor: prev_pbr_mr.base_color_factor().map(|v| v as F),
            base_color_texture: prev_pbr_mr.base_color_texture().map(|i| TexInfo {
                uv_channel: i.tex_coord(),
                texture: images[i.texture().source().index()].clone().into(),
                mime_type: match i.texture().source().source() {
                    gltf::image::Source::View { mime_type, .. } => Some(String::from(mime_type)),
                    gltf::image::Source::Uri { mime_type, .. } => mime_type.map(String::from),
                },
            }),
        };
        out.materials.push(new_mat);
    }

    for anim in doc.animations() {
        let mut samplers = vec![Sampler::default(); anim.samplers().count()];
        let channels = anim
            .channels()
            .map(|c| {
                let target_node_idx = c.target().node().index();
                use crate::anim::Property;
                use gltf::animation::Property as GLTFProperty;
                let target_property = match c.target().property() {
                    GLTFProperty::Translation => Property::Translation(Dim::XYZ),
                    GLTFProperty::Rotation => Property::Rotation(Dim::XYZ),
                    GLTFProperty::Scale => Property::Scale(Dim::XYZ),
                    GLTFProperty::MorphTargetWeights => Property::MorphTargetWeights,
                };
                let sampler = c.sampler().index();
                let reader = c.reader(|buffer: gltf::Buffer| {
                    buffers.get(buffer.index()).map(|data| &data[..])
                });
                let inputs = match reader.read_inputs() {
                    None => vec![],
                    Some(i) => i.map(|f| f as F).collect::<Vec<_>>(),
                };
                use gltf::animation::util::ReadOutputs;
                let outputs = match reader.read_outputs() {
                    None => OutputProperty::None,
                    Some(ReadOutputs::Translations(xyz)) => OutputProperty::Translation(
                        xyz.map(|vs| vs.map(|v| v as F)).collect::<Vec<_>>(),
                    ),
                    Some(ReadOutputs::Scales(xyz)) => {
                        OutputProperty::Scale(xyz.map(|vs| vs.map(|v| v as F)).collect::<Vec<_>>())
                    }
                    Some(ReadOutputs::Rotations(xyz)) => OutputProperty::Rotation(
                        xyz.into_f32()
                            .map(|vs| vs.map(|v| v as F))
                            .collect::<Vec<_>>(),
                    ),
                    Some(ReadOutputs::MorphTargetWeights(w)) => OutputProperty::MorphTargetWeight(
                        w.into_f32().map(|v| v as F).collect::<Vec<_>>(),
                    ),
                };
                samplers[sampler]
                    .input
                    .extend(inputs.iter().copied().map(Time::Float));
                samplers[sampler].output = outputs;
                use gltf::animation::Interpolation as GLTFInterpolation;
                samplers[sampler].interpolation_kind = match c.sampler().interpolation() {
                    GLTFInterpolation::Linear => InterpolationKind::Linear,
                    GLTFInterpolation::CubicSpline => InterpolationKind::CubicSpline,
                    GLTFInterpolation::Step => InterpolationKind::Step,
                };
                Channel {
                    target_node_idx,
                    target_property,
                    sampler: Samplers::One(sampler),
                }
            })
            .collect::<Vec<_>>();

        out.animations.push(Animation {
            name: anim.name().unwrap_or("").to_string(),
            samplers,
            channels,
        });
    }
    for (i, node) in doc.nodes().enumerate() {
        assert_eq!(node.index(), i);
        let mut new_node = GLTFNode::default();
        new_node.index = node.index();
        new_node.name = node.name().unwrap_or("").into();
        new_node.transform = node.transform().matrix().map(|col| col.map(|v| v as F));
        new_node.children = node.children().map(|v| v.index()).collect();

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
            new_skin.skeleton = s.skeleton().map(|j| j.index());
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

                // add in material index
                let mat = p.material();
                if let Some(idx) = mat.index() {
                    let curr_f = new_mesh.f.len();
                    match new_mesh.face_mat_idx.last().cloned() {
                        Some(p) if p.1 == idx && p.0.end == curr_f => {
                            new_mesh.face_mat_idx.last_mut().unwrap().0.end += 1;
                        }
                        None | Some(_) => new_mesh.face_mat_idx.push((curr_f..curr_f + 1, idx)),
                    }
                }
            }
            new_node.mesh = Some(out.meshes.len());
            out.meshes.push(new_mesh);
        }
        out.nodes.push(new_node);
    }
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
                v: mesh.v[i].map(to_f32),
                uv: mesh.uv[0].get(i).copied().unwrap_or_default().map(to_f32),
                n: mesh.n.get(i).copied().unwrap_or_default().map(to_f32),
                joint_idxs: mesh.joint_idxs.get(i).copied().unwrap_or_default(),
                joint_weights: mesh
                    .joint_weights
                    .get(i)
                    .copied()
                    .unwrap_or_default()
                    .map(to_f32),
            };
            bytes.extend(v.to_bytes());
            verts.push(v);
        }
    }

    let mut inv_bind_mat_offsets = vec![];
    for skin in &scene.skins {
        inv_bind_mat_offsets.push(bytes.len());
        for &ibm in &skin.inv_bind_matrices {
            let raw_bytes = ibm.iter().flat_map(|col| {
                col.iter()
                    .flat_map(|&v| to_f32(v).to_le_bytes().into_iter())
            });
            bytes.extend(raw_bytes);
        }
    }

    let f_offset = bytes.len();
    let mut vertex_offset = 0;
    let mut f_offsets = vec![];
    for mesh in &scene.meshes {
        f_offsets.push(bytes.len() - f_offset);
        for f in &mesh.f {
            for tri in f.as_triangle_fan() {
                let raw = tri
                    .into_iter()
                    .flat_map(|vi| (vi as u32 + vertex_offset).to_le_bytes());
                bytes.extend(raw);
            }
        }
        vertex_offset += mesh.v.len() as u32;
    }

    let a_offset = bytes.len();
    let mut a_in_offsets = vec![];
    let mut a_out_offsets = vec![];
    for anim in &scene.animations {
        a_in_offsets.push(vec![]);
        a_out_offsets.push(vec![]);
        for sampler in &anim.samplers {
            a_in_offsets
                .last_mut()
                .unwrap()
                .push(bytes.len() - a_offset);
            let input_bytes = sampler
                .input
                .iter()
                .flat_map(|&v| to_f32(v.to_float()).to_le_bytes());
            bytes.extend(input_bytes);

            a_out_offsets
                .last_mut()
                .unwrap()
                .push(bytes.len() - a_offset);
            use crate::anim::OutputProperty;
            match &sampler.output {
                OutputProperty::None => {}
                OutputProperty::Translation(t) | OutputProperty::Scale(t) => {
                    let raw = t
                        .iter()
                        .flat_map(|p| p.iter().flat_map(|&v| to_f32(v).to_le_bytes()));
                    bytes.extend(raw)
                }
                OutputProperty::Rotation(t) => {
                    let raw = t
                        .iter()
                        .flat_map(|p| p.iter().flat_map(|&v| to_f32(v).to_le_bytes()));
                    bytes.extend(raw)
                }
                OutputProperty::MorphTargetWeight(t) => {
                    let raw = t.iter().flat_map(|&v| to_f32(v).to_le_bytes());
                    bytes.extend(raw)
                }
                // TODO convert these to GLTF properties.
                _ => todo!(),
            }
        }
    }

    let [lb, ub] = verts
        .iter()
        .fold([[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]], |[l, h], n| {
            [
                std::array::from_fn(|i| l[i].min(n.v[i])),
                std::array::from_fn(|i| h[i].max(n.v[i])),
            ]
        });

    // TODO write images here

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
        byte_length: USize64::from(inv_bind_mat_offsets.first().copied().unwrap_or(f_offset)),
        byte_offset: None,
        byte_stride: Some(gltf_json::buffer::Stride(mem::size_of::<Vertex>())),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(gltf_json::buffer::Target::ArrayBuffer)),
    });
    let ibm_buffer_view = if inv_bind_mat_offsets.is_empty() {
        gltf_json::Index::new(0)
    } else {
        root.push(gltf_json::buffer::View {
            buffer,
            byte_length: USize64::from(f_offset - inv_bind_mat_offsets[0]),
            byte_stride: None,
            byte_offset: Some(USize64::from(inv_bind_mat_offsets[0])),
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: None,
        })
    };
    let idx_buffer_view = root.push(gltf_json::buffer::View {
        buffer,
        byte_length: USize64::from(a_offset - f_offset),
        byte_offset: Some(USize64::from(f_offset)),
        byte_stride: None,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(Valid(gltf_json::buffer::Target::ElementArrayBuffer)),
    });

    let anim_buffer_view = if scene.animations.is_empty() {
        gltf_json::Index::new(0)
    } else {
        root.push(gltf_json::buffer::View {
            buffer,
            byte_length: USize64::from(buf_len - a_offset),
            byte_offset: Some(USize64::from(a_offset)),
            byte_stride: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: None,
        })
    };

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
    let inv_bind_matrices = inv_bind_mat_offsets
        .iter()
        .enumerate()
        .map(|(i, &offset)| {
            root.push(gltf_json::Accessor {
                buffer_view: Some(ibm_buffer_view),
                byte_offset: Some(USize64::from(offset - inv_bind_mat_offsets[0])),
                count: USize64::from(scene.skins[i].inv_bind_matrices.len()),
                component_type: Valid(gltf_json::accessor::GenericComponentType(
                    gltf_json::accessor::ComponentType::F32,
                )),
                extensions: Default::default(),
                extras: Default::default(),
                type_: Valid(gltf_json::accessor::Type::Mat4),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            })
        })
        .collect::<Vec<_>>();

    let mut meshes = vec![];
    for (mi, mesh) in scene.meshes.iter().enumerate() {
        let faces = root.push(gltf_json::Accessor {
            buffer_view: Some(idx_buffer_view),
            byte_offset: Some(USize64::from(f_offsets[mi])),
            count: USize64::from(mesh.num_tris() * 3),
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
        meshes.push(mesh);
    }

    let mut inv_bind_idx = 0;
    let skins = scene
        .skins
        .iter()
        .map(|skin| {
            assert_eq!(skin.inv_bind_matrices.len(), skin.joints.len());
            root.push(gltf_json::Skin {
                inverse_bind_matrices: if skin.inv_bind_matrices.is_empty() {
                    None
                } else {
                    let i = inv_bind_matrices[inv_bind_idx];
                    inv_bind_idx += 1;
                    Some(i)
                },
                joints: skin
                    .joints
                    .iter()
                    .map(|&v| gltf_json::Index::new(v as u32))
                    .collect(),
                skeleton: skin.skeleton.map(|s| gltf_json::Index::new(s as u32)),
                name: (!skin.name.is_empty()).then(|| skin.name.clone()),
                extensions: Default::default(),
                extras: Default::default(),
            })
        })
        .collect::<Vec<_>>();

    for (ai, anim) in scene.animations.iter().enumerate() {
        use crate::anim::Property;
        use gltf_json::animation::Property as GLTFProp;
        let channels = anim
            .channels
            .iter()
            .map(|c| gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(c.sampler.one().unwrap() as u32),
                target: gltf_json::animation::Target {
                    node: gltf_json::Index::new(c.target_node_idx as u32),
                    path: Valid(match c.target_property {
                        Property::Translation(_) => GLTFProp::Translation,
                        Property::Rotation(_) => GLTFProp::Rotation,
                        Property::Scale(_) => GLTFProp::Scale,
                        Property::MorphTargetWeights => GLTFProp::MorphTargetWeights,
                    }),
                    extras: Default::default(),
                    extensions: Default::default(),
                },
                extras: Default::default(),
                extensions: Default::default(),
            })
            .collect::<Vec<_>>();
        use crate::anim::InterpolationKind::*;
        use gltf_json::animation::Interpolation as GLTFInterp;
        let samplers = anim
            .samplers
            .iter()
            .enumerate()
            .map(|(si, s)| gltf_json::animation::Sampler {
                input: root.push(gltf_json::Accessor {
                    buffer_view: Some(anim_buffer_view),
                    byte_offset: Some(USize64::from(a_in_offsets[ai][si])),
                    count: USize64::from(s.input.len()),
                    component_type: Valid(gltf_json::accessor::GenericComponentType(
                        gltf_json::accessor::ComponentType::F32,
                    )),
                    extensions: Default::default(),
                    extras: Default::default(),
                    type_: Valid(gltf_json::accessor::Type::Scalar),
                    min: None,
                    max: None,
                    name: None,
                    normalized: false,
                    sparse: None,
                }),
                output: root.push(gltf_json::Accessor {
                    buffer_view: Some(anim_buffer_view),
                    byte_offset: Some(USize64::from(a_out_offsets[ai][si])),
                    count: USize64::from(s.output.len()),
                    component_type: Valid(gltf_json::accessor::GenericComponentType(
                        gltf_json::accessor::ComponentType::F32,
                    )),
                    extensions: Default::default(),
                    extras: Default::default(),
                    type_: Valid(match s.output {
                        OutputProperty::None | OutputProperty::MorphTargetWeight(_) => {
                            gltf_json::accessor::Type::Scalar
                        }
                        OutputProperty::Rotation(_) => gltf_json::accessor::Type::Vec4,
                        OutputProperty::Translation(_) | OutputProperty::Scale(_) => {
                            gltf_json::accessor::Type::Vec3
                        }
                        // Convert these to GLTF accessor types.
                        _ => todo!(),
                    }),
                    min: None,
                    max: None,
                    name: None,
                    normalized: false,
                    sparse: None,
                }),
                interpolation: Valid(match s.interpolation_kind {
                    Linear => GLTFInterp::Linear,
                    CubicSpline => GLTFInterp::CubicSpline,
                    Step => GLTFInterp::Step,
                }),
                extras: Default::default(),
                extensions: Default::default(),
            })
            .collect::<Vec<_>>();
        let a = gltf_json::Animation {
            channels,
            samplers,
            extensions: Default::default(),
            extras: Default::default(),
            name: (!anim.name.is_empty()).then(|| anim.name.clone()),
        };
        root.push(a);
    }

    let mut nodes = vec![];
    for (ni, n) in scene.nodes.iter().enumerate() {
        let matrix = if n.transform.is_identity() {
            None
        } else {
            Some(unsafe {
                std::mem::transmute::<[[f32; 4]; 4], [f32; 16]>(
                    n.transform.to_mat().map(|col| col.map(to_f32)),
                )
            })
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
            mesh: n.mesh.map(|mi| meshes[mi]),
            name: (!n.name.is_empty()).then(|| n.name.clone()),

            skin: n.skin.map(|s| skins[s]),
            children,
            matrix,
            ..Default::default()
        });
        assert_eq!(node.value(), ni);

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
    let scene = load("etrian_odyssey_3_monk.glb").unwrap();
    assert_eq!(scene.root_nodes.len(), 1);
    assert_eq!(scene.meshes.len(), 24);
    assert_eq!(scene.nodes.len(), 293);
    assert_eq!(scene.skins.len(), 18);
}

#[test]
fn test_gltf_load_save() {
    let scene = load("etrian_odyssey_3_monk.glb").expect("Failed to open");
    let out = std::fs::File::create("gltf_test.glb").expect("Failed to create file");
    save_glb(&crate::mesh::Scene::from(scene), io::BufWriter::new(out))
        .expect("Failed to write file");
    todo!();
}

#[test]
fn test_simple_skin_load_save() {
    let scene = load("simple_skin.gltf").expect("Failed to open");
    assert_eq!(scene.nodes.len(), 3);
    assert_eq!(scene.skins.len(), 1);
    let out = std::fs::File::create("simple_skin_io.glb").expect("Failed to create file");
    save_glb(&crate::mesh::Scene::from(scene), io::BufWriter::new(out))
        .expect("Failed to write file");
    todo!();
}
