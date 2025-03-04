#![allow(unused)]

use super::parser::{Data, Token, KV};
use super::{id, FBXAnimCurveNode, FBXAnimLayer, FBXCluster, FBXMesh, FBXNode, FBXScene, FBXSkin};
use crate::F;
use std::io::{self, Seek, SeekFrom, Write};

use std::collections::{HashMap, HashSet};

macro_rules! push_kv {
    ($kvs: expr, $kv: expr) => {{
        let idx = $kvs.len();
        $kvs.push($kv);
        idx
    }};
}

macro_rules! add_kvs {
  ($kvs: expr, $parent: expr
    $(, $(if $cond: expr =>)? $field: literal, $values: expr $( => $children_func: expr )? )* $(,)?
  ) => {{
    $($( if $cond )? {
      let c = push_kv!($kvs, KV {
        key: $field.to_string(),
        values: $values.to_vec(),
        parent: Some($parent),
      });
      $( $children_func(c); )?
    })*
  }}
}

macro_rules! root_fields {
  ($kvs: ident, $key: expr, $values: expr $( => $children_func: expr )? $(,)?) => {{
    // Sometimes used
    let _root_id = push_kv!($kvs, KV {
      key: $key.to_string(),
      values: $values.to_vec(),
      parent:None
    });
    $( $children_func(_root_id); )?
  }}
}

macro_rules! object_to_kv {
    ($parent: expr, $kind: expr, $id: expr, $name: expr, $obj_type: expr, $classtag: expr $(,)?) => {{
        let vals = [
            Data::I64($id as i64),
            Data::String(format!("{}\x00\x01{}", $name, $obj_type)),
            Data::str($classtag),
        ];

        KV::new($kind, &vals, $parent)
    }};
}

macro_rules! conn_oo {
    ($conn_idx: expr, $src: expr, $dst: expr) => {{
        let vals = &[
            Data::str("OO"),
            Data::I64($src as i64),
            Data::I64($dst as i64),
        ];
        KV::new("C", vals, Some($conn_idx))
    }};
}

// 1. convert scene to KVs
// 2. convert KVs to tokens
// 3. export tokens as binary to writer
pub fn export_fbx(scene: &FBXScene, w: (impl Write + Seek)) -> io::Result<()> {
    let kvs = scene.to_kvs();
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut roots = vec![];
    for (kvi, kv) in kvs.iter().enumerate() {
        let Some(p) = kv.parent else {
            roots.push(kvi);
            continue;
        };
        children.entry(p).or_default().push(kvi);
    }

    let mut token_sets = vec![];
    let mut seen = HashSet::new();
    for i in roots {
        let mut tokens = vec![];
        tokenize(&kvs, i, &children, &mut tokens, &mut seen);
        token_sets.push(tokens);
    }

    write_tokens(&token_sets, w)
}

pub fn i32_p(name: &str, val: i32) -> [Data; 5] {
    [
        Data::str(name),
        Data::str("int"),
        Data::str("Integer"),
        Data::str(""),
        Data::I32(val),
    ]
}
pub fn f64_p(name: &str, val: f64) -> [Data; 5] {
    [
        Data::str(name),
        Data::str("double"),
        Data::str("Number"),
        Data::str(""),
        Data::F64(val),
    ]
}

impl FBXScene {
    pub(crate) fn to_kvs(&self) -> Vec<KV> {
        let mut kvs = vec![];

        // have ordered the root fields to match the order of input

        root_fields!(
            kvs,
            "FBXHeaderExtension", &[] => |c| add_kvs!(
              kvs, c,
              "FBXVersion", &[Data::I32(7600)],
              "FBXHeaderVersion", &[Data::I32(1003)],
              "EncryptionType", &[Data::I32(0)],
              "Creator", &[Data::str("pars3d")],
              "CreationTimeStamp", &[] => |c| add_kvs!(
                kvs, c,
                "Version", &[Data::I32(101)],
                "Year", &[Data::I32(1970)],
                "Month", &[Data::I32(1)],
                "Day", &[Data::I32(1)],
                "Hour", &[Data::I32(1)],
                "Minute", &[Data::I32(1)],
                "Second", &[Data::I32(1)],
                "Millisecond", &[Data::I32(1)],
              ),
              "SceneInfo", &[Data::str("GlobalInfo\x00\x01SceneInfo"), Data::str("UserData")]
              => |c| add_kvs!(
                kvs, c,
                "Type", &[Data::str("UserData")],
                "Version", &[Data::I32(100)],
                "MetaData", &[] => |c| add_kvs!(
                  kvs, c,
                  "Version", &[Data::I32(100)],
                  "Title", &[Data::str("")],
                  "Subject", &[Data::str("")],
                  "Author", &[Data::str("")],
                  "Keywords", &[Data::str("")],
                  "Revision", &[Data::str("")],
                  "Comment", &[Data::str("")],
                ),
                "Properties70", &[] => |_| {},
              ),
            ),
        );
        root_fields!(kvs, "FileId", &[Data::Binary(self.file_id.to_vec())]);
        root_fields!(kvs, "CreationTime", &[Data::str("1970-01-01 10:00:00:000")]);
        root_fields!(kvs, "Creator", &[Data::str("pars3d")]);

        let settings = &self.global_settings;
        root_fields!(
          kvs,
          "GlobalSettings", &[] => |v| add_kvs!(
            kvs, v,
            "Version", &[Data::I32(1000)],
            "Properties70", &[] => |v| add_kvs!(kvs, v,
              "P", i32_p("UpAxis", settings.up_axis),
              "P", i32_p("UpAxisSign", settings.up_axis_sign),
              "P", i32_p("FrontAxis", settings.front_axis),
              "P", i32_p("FrontAxisSign", settings.front_axis_sign),
              "P", i32_p("CoordAxis", settings.coord_axis),
              "P", i32_p("CoordAxisSign", settings.coord_axis_sign),
              "P", i32_p("OriginalUpAxis", settings.og_up_axis),
              "P", i32_p("OriginalUpAxisSign", settings.og_up_axis_sign),
              "P", f64_p("UnitScaleFactor", settings.unit_scale_factor),
              "P", f64_p("OriginalUnitScaleFactor", settings.og_unit_scale_factor),
            ),
          ),
        );

        root_fields!(
          kvs,
          "Documents", &[] => |c| add_kvs!(
            kvs, c,
            "Count", &[Data::I32(1)],
            "Document", &[Data::I64(self.id as i64), Data::str("Scene"), Data::str("Scene")]
            => |c| add_kvs!(
              kvs, c,
              "RootNode", &[Data::I64(0)], /* TODO this should read from something else */
              "Properties70", &[],
            ),
          ),
        );

        root_fields!(kvs, "References", &[]);

        root_fields!(
          kvs, "Definitions", &[] => |c| add_kvs!(kvs, c,
            "Version", &[Data::I32(101)],
            "Count", &[Data::I32(1 /* not sure what this is */)],
            "ObjectType", &[Data::str("" /* not sure */)] => |c| add_kvs!(
                kvs, c,
                "Count", &[Data::I32(1)],
                "PropertyTemplate", &[Data::str("FbxMesh")]
            ),
          ),
        );

        let obj_kv = push_kv!(kvs, KV::new("Objects", &[], None));
        let conn_idx = push_kv!(kvs, KV::new("Connections", &[], None));

        for mesh in &self.meshes {
            mesh.to_kvs(obj_kv, &mut kvs);
        }

        for node in &self.nodes {
            node.to_kvs(obj_kv, &mut kvs);

            if let Some(ln) = node.limb_node(obj_kv, &mut kvs) {
                push_kv!(kvs, conn_oo!(conn_idx, ln, node.id));
            }
        }

        for skin in &self.skins {
            skin.to_kvs(obj_kv, &mut kvs);

            push_kv!(kvs, conn_oo!(conn_idx, skin.id, self.meshes[skin.mesh].id));
        }

        for cl in &self.clusters {
            cl.to_kvs(obj_kv, &mut kvs);

            let node = &self.nodes[cl.node];
            push_kv!(kvs, conn_oo!(conn_idx, cl.id, self.skins[cl.skin].id));
            // not sure what ID this is going to
            push_kv!(kvs, conn_oo!(conn_idx, node.id, cl.id));
        }

        // for each node add a connection from it to its parent
        for ni in 0..self.nodes.len() {
            let parent = self.parent_node(ni);
            let id = match parent {
                None => 0,
                Some(p) => self.nodes[p].id,
            };
            let own_id = self.nodes[ni].id;
            push_kv!(kvs, conn_oo!(conn_idx, own_id, id));
        }

        // also add connections from nodes to their mesh
        for node in &self.nodes {
            let Some(mi) = node.mesh else {
                continue;
            };

            push_kv!(kvs, conn_oo!(conn_idx, self.meshes[mi].id, node.id));
        }

        root_fields!(
          kvs,
          "Takes", &[] => |c| add_kvs!(
            kvs, c, "Current", &[Data::str("")]
          )
        );

        kvs
    }
}

impl FBXMesh {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let mesh_kv = object_to_kv!(
            Some(parent),
            "Geometry",
            self.id,
            self.name,
            "Geometry",
            "Mesh"
        );
        let mesh_kv = push_kv!(kvs, mesh_kv);

        let to_f64 = |&v| v as f64;
        let to_i32 = |&v| v as i32;
        let vert_vals = self
            .v
            .iter()
            .flat_map(|v| v.iter().map(to_f64))
            .collect::<Vec<f64>>();

        let faces = self
            .f
            .iter()
            .flat_map(|f| {
                let (&last, rest) = f.as_slice().split_last().unwrap();
                let last = last as i32;
                rest.iter().map(to_i32).chain(std::iter::once(-last - 1))
            })
            .collect::<Vec<i32>>();

        add_kvs!(
            kvs,
            mesh_kv,
            "Properties70", &[],
            "GeometryVersion", &[Data::I32(101)],
            "Vertices", &[Data::F64Arr(vert_vals)],
            "PolygonVertexIndex", &[Data::I32Arr(faces)],
            /*
            "Layer", &[Data::I32(0)] => |c| add_kvs!(
              kvs, c,
              "Version", &[Data::I32(100)],
              "LayerElement", &[] => |c| add_kvs!(
                kvs, c,
                "Type", &[Data::str("LayerElementNormal")],
                "TypedIndex", &[Data::I32(0)],
              ),
            ),
            */

            if !self.n.is_empty() => "LayerElementNormal", &[Data::I32(0)].as_slice() => |c| add_kvs!(
              kvs, c,
              "Version", &[Data::I32(101)],
              "Name", &[Data::str("Normal1")],
              "MappingInformationType", &[Data::str(self.n.map_kind.to_str())],
              "ReferenceInformationType", &[Data::str(self.n.ref_kind.to_str())],
              "Normals", &[Data::F64Arr(
                self.n.values.iter().flat_map(|n| n.iter().map(to_f64)).collect::<Vec<_>>()
              )],
              if !(self.n.map_kind.is_by_vertices() && self.n.ref_kind.is_direct()) =>
                "NormalsIndex", &[Data::I32Arr(self.n.indices.iter().map(to_i32).collect::<Vec<_>>())],
            ),

            if !self.uv.is_empty() => "LayerElementUV", &[Data::I32(0)] => |c| add_kvs!(
              kvs, c,
              "Version", &[Data::I32(101)],
              "Name", &[Data::str("UV0")],
              "MappingInformationType", &[Data::str(self.uv.map_kind.to_str())],
              "ReferenceInformationType", &[Data::str(self.uv.ref_kind.to_str())],
              "UV", &[Data::F64Arr(
                self.uv.values.iter().flat_map(|uv| uv.iter().map(to_f64)).collect::<Vec<_>>()
              )],
              if !(self.uv.map_kind.is_by_vertices() && self.uv.ref_kind.is_direct()) =>
                "UVIndex", &[Data::I32Arr(self.uv.indices.iter().map(to_i32).collect::<Vec<_>>())],
            ),
        );
    }
}

impl FBXNode {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let tform_part = |n, vs: [F; 3]| {
            [
                Data::str(n),
                Data::str(n),
                Data::str(""),
                Data::str("A"),
                Data::F64(vs[0] as f64),
                Data::F64(vs[1] as f64),
                Data::F64(vs[2] as f64),
            ]
        };

        let node_kv = object_to_kv!(
            Some(parent),
            "Model",
            self.id,
            self.name,
            "Model",
            if self.mesh.is_some() {
                "Mesh"
            } else if self.limb_node_id.is_some() {
                "LimbNode"
            } else {
                "Null"
            }
        );
        let node_kv = push_kv!(kvs, node_kv);
        add_kvs!(
            kvs, node_kv,
            "Version", &[Data::I32(101)],
            "Properties70", &[] => |c| add_kvs!(kvs, c,
              if self.transform.rotation != [0.; 3] => "P", tform_part("Lcl Rotation", self.transform.rotation),
              if self.transform.scale != [1.; 3] => "P", tform_part("Lcl Scaling", self.transform.scale),
              if self.transform.translation != [0.; 3] => "P", tform_part("Lcl Translation", self.transform.translation),
              if self.hidden => "P", [
                Data::str("Visibility"), Data::str("Visibility"), Data::str(""), Data::str("A"), Data::F64(0.0)
              ],
            ), /* children properties */
            "Culling", &[Data::str("CullingOff")],
            "MultiTake", &[Data::I32(0)],
            "MultiLayer", &[Data::I32(0)],
            "Shading", &[Data::Bool(false)],
        );
    }

    fn limb_node(&self, parent: usize, kvs: &mut Vec<KV>) -> Option<usize> {
        let limb_node_id = self.limb_node_id?;
        let limb_node_kv = object_to_kv!(
            Some(parent),
            "NodeAttribute",
            limb_node_id,
            self.name,
            "NodeAttribute",
            "LimbNode"
        );
        let limb_node_kv = push_kv!(kvs, limb_node_kv);
        add_kvs!(kvs, limb_node_kv, "Version", &[Data::I32(101)],);

        Some(limb_node_id)
    }
}

impl FBXSkin {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let skin_kv = object_to_kv!(
            Some(parent),
            "Deformer",
            self.id,
            self.name,
            "Deformer",
            "Skin"
        );
        let skin_kv = push_kv!(kvs, skin_kv);
        add_kvs!(
            kvs,
            skin_kv,
            "Version",
            &[Data::I32(101)],
            "SkinningType",
            &[Data::str("Linear")],
        );
    }
}

impl FBXCluster {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let cl_kv = object_to_kv!(
            Some(parent),
            "Deformer",
            self.id,
            self.name,
            "SubDeformer",
            "Cluster"
        );
        let cl_kv = push_kv!(kvs, cl_kv);
        add_kvs!(
            kvs,
            cl_kv,
            "Version",
            &[Data::I32(101)],
            "Indexes",
            &[Data::I32Arr(
                self.indices
                    .iter()
                    .map(|&idx| idx as i32)
                    .collect::<Vec<_>>()
            )],
            "Weights",
            &[Data::F64Arr(
                self.weights.iter().map(|&w| w as f64).collect::<Vec<_>>()
            )],
            if self.tform != [[0.; 4]; 4] => "Transform",
            &[Data::F64Arr(
                self.tform
                    .iter()
                    .flat_map(|&v| v.into_iter())
                    .map(|v| v as f64)
                    .collect::<Vec<_>>()
            )],
            if self.tform_link != [[0.; 4]; 4] => "TransformLink",
            &[Data::F64Arr(
                self.tform_link
                    .iter()
                    .flat_map(|&v| v.into_iter())
                    .map(|v| v as f64)
                    .collect::<Vec<_>>()
            )],
            if self.tform_assoc_model != [[0.; 4]; 4] => "TransformAssociateModel", &[Data::F64Arr(
              self.tform_assoc_model.iter().flat_map(|&v| v.into_iter())
                .map(|v| v as f64)
                .collect::<Vec<_>>()
            )],
        );
    }
}

impl FBXAnimLayer {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let al_kv = object_to_kv!(
            Some(parent),
            "AnimationLayer",
            self.id,
            self.name,
            "AnimLayer",
            ""
        );
        let al_kv = push_kv!(kvs, al_kv);
    }
}

impl FBXAnimCurveNode {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let acn_kv = object_to_kv!(
            Some(parent),
            "AnimationCurveNode",
            self.id,
            self.name,
            "AnimCurveNode",
            ""
        );
        let acn_kv = push_kv!(kvs, acn_kv);
        add_kvs!(kvs, acn_kv, "Properties70", &[]);
    }
}

fn tokenize(
    kvs: &[KV],
    curr: usize,
    children: &HashMap<usize, Vec<usize>>,
    tokens: &mut Vec<Token>,

    seen: &mut HashSet<usize>,
) {
    assert!(seen.insert(curr), "{seen:?} {curr:?}");
    let kv = &kvs[curr];
    tokens.push(Token::Key(kv.key.clone()));
    for v in &kv.values {
        tokens.push(Token::Data(v.clone()));
    }

    let no_kids = vec![];
    let curr_children = children.get(&curr).unwrap_or(&no_kids);
    if curr_children.is_empty() {
        return;
    }
    tokens.push(Token::ScopeStart);
    for &c in curr_children {
        tokenize(kvs, c, children, tokens, seen);
    }
    tokens.push(Token::ScopeEnd);
}

pub fn write_token_set(
    tokens: &[Token],
    offset: usize,
    w: &mut (impl Write + Seek),
) -> io::Result<(usize, usize)> {
    assert!(!tokens.is_empty());
    let mut written = 0;
    macro_rules! write_word {
        ($dst:expr, $word: expr) => {{
            write_word!($dst, u64, $word)
        }};
        ($dst:expr, bool, $w: expr) => {{
            write_word!($dst, u8, if $w { 1 } else { 0 })
        }};
        ($dst:expr, $ty: ty, $w: expr) => {{
            $dst.write(&($w as $ty).to_le_bytes())?
        }};
    }
    macro_rules! write_arr {
        ($dst:expr, $ty: ty, $arr: expr) => {{
            let len = $arr.len();
            let mut w = write_word!($dst, u32, len);
            // TODO compress
            w += write_word!($dst, u32, 0);
            w += write_word!($dst, u32, len * std::mem::size_of::<$ty>());
            for v in $arr {
                w += write_word!($dst, $ty, *v);
            }
            w
        }};
    }
    macro_rules! write_string {
        ($dst:expr, $str: expr, $is_long: expr, $allow_null: expr) => {{
            let c = if $is_long {
                write_word!($dst, u32, $str.len())
            } else {
                write_word!($dst, u8, $str.len())
            };
            assert!($allow_null || $str.as_bytes().iter().all(|&v| v != b'\0'));
            c + $dst.write($str.as_bytes())?
        }};
    }

    macro_rules! write_data {
        ($dst: expr, $d: expr) => {{
            let c = match $d {
                Data::I16(_) => 'Y',

                Data::I32(_) => 'I',
                Data::I32Arr(_) => 'i',

                Data::I64(_) => 'L',
                Data::I64Arr(_) => 'l',

                Data::F32(_) => 'F',
                Data::F32Arr(_) => 'f',

                Data::F64(_) => 'D',
                Data::F64Arr(_) => 'd',

                Data::Binary(_) => 'R',
                Data::String(_) => 'S',
                Data::Bool(_) => 'C',
                x => todo!("{x:?}"),
            };
            let c = write_word!($dst, u8, c);
            assert_eq!(c, 1);
            c + match $d {
                Data::I32(i) => write_word!($dst, i32, *i),
                Data::I64(i) => write_word!($dst, i64, *i),
                Data::F64(i) => write_word!($dst, f64, *i),
                Data::I64Arr(arr) => write_arr!($dst, i64, arr),
                Data::I32Arr(arr) => write_arr!($dst, i32, arr),
                Data::F64Arr(arr) => write_arr!($dst, f64, arr),
                Data::String(s) => write_string!($dst, s, true, true),
                Data::Bool(b) => write_word!($dst, bool, *b),
                Data::Binary(b) => {
                    let len = write_word!($dst, u32, b.len());
                    assert_eq!(len, 4);
                    let v = $dst.write(&b)?;
                    assert_eq!(v, b.len());
                    len + v
                }
                x => todo!("{x:?}"),
            }
        }};
    }

    let Token::Key(k) = &tokens[0] else {
        panic!("{:?}", tokens[0]);
    };
    let mut i = 1;
    let mut prop_count = 0;
    let mut prop_len = 0;
    while i + prop_count < tokens.len()
        && let Token::Data(ref d) = &tokens[i + prop_count]
    {
        prop_len += write_data!(std::io::sink(), d);
        prop_count += 1;
    }
    const OFFSET_PLACEHOLDER: u64 = u64::MAX;
    let pos_to_write = (written + offset) as u64;
    // offset
    written += write_word!(w, OFFSET_PLACEHOLDER);
    // prop count
    written += write_word!(w, prop_count);
    // prop len
    written += write_word!(w, prop_len);
    // Scope name
    written += write_string!(w, k, false, false);

    for j in 0..prop_count {
        let Token::Data(ref d) = &tokens[i + j] else {
            panic!();
        };
        written += write_data!(w, d);
    }
    i += prop_count;
    const SENTINEL_BLOCK_LEN: usize = size_of::<u64>() * 3 + 1;
    const SENTINEL: [u8; SENTINEL_BLOCK_LEN] = [b'\0'; SENTINEL_BLOCK_LEN];
    assert!(i >= tokens.len() || !matches!(tokens[i], Token::Data(_)));
    if i < tokens.len() && tokens[i] == Token::ScopeStart {
        i += 1;
        while i < tokens.len() && tokens[i] != Token::ScopeEnd {
            let (wrote, tkns) = write_token_set(&tokens[i..], offset + written, w)?;
            i += tkns;
            written += wrote;
        }
        if i < tokens.len() {
            assert_eq!(tokens[i], Token::ScopeEnd);
            i += 1;
        }
        assert_eq!(w.write(&SENTINEL)?, SENTINEL_BLOCK_LEN);
        written += SENTINEL_BLOCK_LEN;
    }
    w.seek(SeekFrom::Start(pos_to_write))?;
    write_word!(w, offset + written);
    w.seek(SeekFrom::End(0))?;

    Ok((written, i))
}

pub fn write_tokens(token_sets: &[Vec<Token>], mut w: (impl Write + Seek)) -> io::Result<()> {
    let mut offset = 0;
    offset += w.write(super::parser::MAGIC)?;
    let version = (7600u32).to_le_bytes();
    offset += w.write(&version)?;
    assert_eq!(offset, super::parser::MAGIC.len() + 4);

    for t in token_sets {
        let (w, tkns) = write_token_set(t, offset, &mut w)?;
        assert_eq!(tkns, t.len());
        offset += w;
    }
    // write an empty prop to signify the end
    let _ = w.write(&(0u64).to_le_bytes())?;
    let _ = w.write(&(0u64).to_le_bytes())?;
    let _ = w.write(&(0u64).to_le_bytes())?;
    let _ = w.write(&(0u8).to_le_bytes())?;

    Ok(())
}
