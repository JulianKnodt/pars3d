use super::F;

#[derive(Default, Clone, PartialEq)]
pub struct GLTFMesh {
    pub v: Vec<[F; 3]>,
    pub uvs: Vec<[F; 2]>,
    pub f: Vec<[usize; 3]>,
    // For each vertex, associate it with 4 bones
    pub joint_idxs: Vec<[u16; 4]>,
    pub joint_weights: Vec<[F; 4]>,
}

pub fn load<P>(path: P) -> gltf::Result<GLTFMesh>
where
    P: AsRef<std::path::Path>,
{
    let (doc, buffers, _images) = gltf::import(path)?;

    let mut out = GLTFMesh::default();

    fn traverse_node(
        gltf: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        node: &gltf::scene::Node,
        out: &mut GLTFMesh,
    ) {
        if let Some(m) = node.mesh() {
            let offset = out.v.len();
            for p in m.primitives() {
                let reader = p.reader(|buffer: gltf::Buffer| {
                    buffers.get(buffer.index()).map(|data| &data[..])
                });
                let mut num_pos = 0;
                for p in reader.read_positions().into_iter().flatten() {
                    out.v.push(p);
                    num_pos += 1;
                }

                out.uvs.extend(
                    reader
                        .read_tex_coords(0)
                        .map(|v| v.into_f32())
                        .into_iter()
                        .flatten(),
                );

                if let Some(jr) = reader.read_joints(0) {
                    let mut num_joint = 0;
                    for p in jr.into_u16() {
                        out.joint_idxs.push(p);
                        num_joint += 1;
                    }
                    assert_eq!(num_pos, num_joint);
                    let Some(jwr) = reader.read_weights(0) else {
                        panic!("has joints but no weights?");
                    };
                    out.joint_weights.extend(jwr.into_f32());
                }

                let idxs = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .array_chunks::<3>();
                for vis in idxs {
                    out.f.push(vis.map(|vi| vi as usize + offset));
                }
            }
        }
        for child in node.children() {
            traverse_node(gltf, buffers, &child, out);
        }
    }

    for scene in doc.scenes() {
        for root_node in scene.nodes() {
            traverse_node(&doc, &buffers, &root_node, &mut out);
        }
    }
    Ok(out)
}

#[test]
fn test_load_gltf() {
    let cuphead = load("cuphead.glb").unwrap();
    /*
    for [x,y,z] in &cuphead.v {
      println!("v {x} {y} {z}");
    }
    for ijk in &cuphead.f {
      let [i,j,k] = ijk.map(|i| i + 1);
      println!("f {i} {j} {k}");
    }
    */
}
