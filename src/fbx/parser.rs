#![allow(unused)]

use super::{FBXMesh, FBXNode, FBXScene, FBXSettings};
use crate::{FaceKind, F};

use std::ascii::Char;
use std::collections::HashMap;
use std::io::{self, BufRead, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::Path;

use std::assert_matches::assert_matches;

/// Magic binary length.
pub(crate) const MAGIC_LEN: usize = 23;

/// Magic binary.
pub(crate) const MAGIC: &[u8; MAGIC_LEN] = b"Kaydara FBX Binary  \x00\x1a\x00";

#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),

    F32(f32),
    F64(f64),

    Bool(bool),
    String(String),
    Binary(Vec<u8>),

    I32Arr(Vec<i32>),
    I64Arr(Vec<i64>),

    F32Arr(Vec<f32>),
    F64Arr(Vec<f64>),
    BoolArr(Vec<bool>),

    /// Unknown how to read this data, has the size in it
    Unknown(usize),

    /// Marker to indicate that data was moved from this
    Used,
}

macro_rules! cast {
    ($fn_name: ident, $out_ty: ty, $variant: tt) => {
        fn $fn_name(&self) -> Option<$out_ty> {
            match self {
                Data::$variant(v) => Some(v),
                _ => None,
            }
        }
    };
}

impl Data {
    cast!(as_str, &str, String);
    cast!(as_f64_arr, &[f64], F64Arr);
    cast!(as_i64_arr, &[i64], I64Arr);
    cast!(as_i32_arr, &[i32], I32Arr);

    cast!(as_i64, &i64, I64);
    cast!(as_i32, &i32, I32);

    cast!(as_f64, &f64, F64);

    fn as_int(&self) -> Option<i64> {
        match *self {
            Data::I64(v) => Some(v),
            Data::I32(v) => Some(v as i64),
            _ => None,
        }
    }
    fn as_float(&self) -> Option<F> {
        match *self {
            Data::F64(v) => Some(v as F),
            Data::F32(v) => Some(v as F),
            _ => None,
        }
    }

    pub fn str(s: &str) -> Self {
        Data::String(String::from(s))
    }
}

/// How to map some information to a mesh
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingKind {
    PerPolygon,
    Uniform,
}

#[derive(Debug, PartialEq)]
pub enum Token {
    Key(String),
    Data(Data),
    ScopeStart,
    ScopeEnd,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct KV {
    pub(crate) key: String,
    pub(crate) values: Vec<Data>,
    pub(crate) parent: Option<usize>,
}

impl KV {
    pub fn new(key: impl Into<String>, values: &[Data], parent: Option<usize>) -> KV {
        let key = key.into();
        let values = values.to_vec();
        KV {
            key,
            values,
            parent,
        }
    }
    pub fn id(&self) -> Option<i64> {
        self.values.first().and_then(Data::as_int)
    }
}

macro_rules! match_children {
  ($self: ident, $i: expr $(, $key:expr, $mtch: pat => $val:expr )* $(,)? ) => {{
    let kv = &$self.kvs[$i];
    for &c in $self.children.get(&$i).map(Vec::as_slice).unwrap_or(&[]) {
      let c_kv = &$self.kvs[c];
      match c_kv.key.as_str() {
        $($key => {
          assert_matches!(c_kv.values.as_slice(), $mtch, "{} {}", $key, kv.key);
          $val(c)
        })*
        x => todo!("Unhandled key {x:?} in {} children", kv.key),
      }
    }
  }};
}

macro_rules! root_fields {
  ($self: ident, $key: expr, $vals: pat
  $(, $sub_field: expr, $mtch: pat => $values_func: expr)* $(,)?) => {{
    if let Some(r) = $self.find_root($key) {
      assert_matches!($self.kvs[r].values.as_slice(), $vals);

      for &c in $self.children.get(&r).map(Vec::as_slice).unwrap_or(&[]) {
        let c_kv = &$self.kvs[c];
        match c_kv.key.as_str() {
          $($sub_field => {
            assert_matches!(c_kv.values.as_slice(), $mtch, "{} {}", $key, c_kv.key);
            $values_func(c);
          })*
          x => todo!("Unhandled {x} in {}", $key),
        }
      }
    } else {
      eprintln!("FBX missing root field {}", $key);
    }
  }}
}

#[derive(Debug, Clone, Default)]
pub struct KVs {
    kvs: Vec<KV>,
    roots: Vec<usize>,
    children: HashMap<usize, Vec<usize>>,
}

impl KVs {
    fn find_root(&self, v: &str) -> Option<usize> {
        self.roots.iter().find(|&&i| self.kvs[i].key == v).copied()
    }
    // parses the token stream until a scope end.
    // returns the index of the newly produced datablock.
    fn parse_scope(&mut self, tokens: &mut impl Iterator<Item = Token>, parent: Option<usize>) {
        let mut i = self.kvs.len();
        self.kvs.push(Default::default());
        self.kvs[i].parent = parent;
        match parent {
            None => self.roots.push(i),
            Some(p) => self.children.entry(p).or_default().push(i),
        }
        while let Some(n) = tokens.next() {
            match n {
                Token::Key(k) => {
                    if self.kvs[i].key.is_empty() {
                        self.kvs[i].key = k;
                    } else {
                        i = self.kvs.len();
                        self.kvs.push(Default::default());
                        self.kvs[i].parent = parent;
                        self.kvs[i].key = k;
                        match parent {
                            None => self.roots.push(i),
                            Some(p) => self.children.entry(p).or_default().push(i),
                        }
                    }
                }
                Token::ScopeStart => self.parse_scope(tokens, Some(i)),
                Token::Data(d) => self.kvs[i].values.push(d),
                Token::ScopeEnd => return,
            }
        }
    }

    /// Constructs a graphviz representation of this FBX file, for viewing externally
    pub fn to_graphviz(&self, mut dst: impl Write) -> io::Result<()> {
        writeln!(dst, "graph FBX {{")?;
        for (i, kv) in self.kvs.iter().enumerate() {
            match kv.id() {
                Some(id) => writeln!(dst, "\t{i} [label=\"{id}, {}\"]", kv.key)?,
                None => writeln!(dst, "\t{i} [label=\"{}\"]", kv.key)?,
            }
        }
        for (k, vs) in self.children.iter() {
            for &v in vs.iter() {
                writeln!(dst, "\t{k} -- {v}")?;
            }
        }
        writeln!(dst, "}}")?;
        Ok(())
    }

    fn parse_node(&self, node_id: i64, kvi: usize) -> FBXNode {
        let mut out = FBXNode::default();
        assert!(node_id >= 0);
        out.id = node_id as usize;

        match_children!(
          self,
          kvi,
          "Version", &[Data::I32(_)] => |v| {},
          "Properties70", &[] => |v| match_children!(
            self,
            v,
            "P", &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::F64(_), Data::F64(_), Data::F64(_)
            ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I32(_),
            ] => |v: usize| {
                let vals = &self.kvs[v].values;
                match vals[0].as_str().unwrap() {
                  "Lcl Rotation" => {},
                  "Lcl Scaling" => {},
                  "DefaultAttributeIndex" => {},
                  "InheritType" => {},
                  x => todo!("{x:?}"),
                }
             },
          ),
          "MultiLayer", &[Data::I32(_)] => |v| {},
          "Culling", &[Data::String(_)] => |v: usize| {
            assert_matches!(self.kvs[v].values[0].as_str(), Some("CullingOff" | "CullingOn"));
          },
          "MultiTake", &[Data::I32(_)] => |v| {},
          "Shading", &[Data::Bool(_)] => |v| {},
        );
        out
    }
    fn parse_mesh(&self, mesh_id: i64, kvi: usize) -> FBXMesh {
        let mut out = FBXMesh::default();
        assert!(mesh_id >= 0);
        out.id = mesh_id as usize;
        match_children!(
          self,
          kvi,
          "Properties70", &[] => |c| match_children!(self, c),
          "Vertices", &[Data::F64Arr(_)] => |c: usize| {
              // TODO can this be f32 arr?
              let v_arr: &[f64] = self.kvs[c].values[0].as_f64_arr().unwrap();
              let v = v_arr
                  .iter()
                  .array_chunks::<3>()
                  .map(|[a, b, c]| [*a as F, *b as F, *c as F]);
              out.v.extend(v);
          },
          "GeometryVersion", &[Data::I32(_)] => |c| match_children!(self, c),
          "PolygonVertexIndex", &[Data::I32Arr(_)] => |c: usize| {
              let mut curr_face = FaceKind::empty();
              let idxs = self.kvs[c].values[0].as_i32_arr().unwrap();
              for &vi in idxs {
                  if vi >= 0 {
                      curr_face.insert(vi as usize);
                  } else {
                      curr_face.insert(-(vi + 1) as usize);
                      let f = std::mem::replace(&mut curr_face, FaceKind::empty());
                      out.f.push(f);
                  }
              }
          },
          "Edges", &[Data::I32Arr(_)] => |c| { /* No idea what to do here */ },
          "LayerElementNormal", &[Data::I32(_/*index of normal*/)] => |c| match_children!(
              self, c,
              "Version", &[Data::I32(_)] => |_| {},
              "Name", &[Data::String(_)] => |_| {},
              "MappingInformationType", &[Data::String(_)] => |c: usize| {
                  assert_eq!(&self.kvs[c].values[0], &Data::str("ByPolygonVertex"));
              },
              "ReferenceInformationType", &[Data::String(_)] => |c: usize| {},
              "Normals", &[Data::F64Arr(_)] => |c: usize| {
                  let gc = &self.kvs[c];
                  let Data::F64Arr(ref arr) = &gc.values[0] else {
                    todo!();
                  };
                  out.n
                      .extend(arr.array_chunks::<3>().map(|n| n.map(|v| v as F)));
              },
              "NormalsIndex", &[Data::I32Arr(_)] => |c: usize| {
                  let gc = &self.kvs[c];
                  let Data::I32Arr(ref arr) = &gc.values[0] else {
                    todo!();
                  };
                  let idxs = arr
                      .iter()
                      .copied()
                      .inspect(|&idx| assert!(idx >= 0))
                      .map(|v| v as usize);
                  out.vert_norm_idx.extend(idxs);
              },
              "NormalsW", &[Data::F64Arr(_)] => |c| { /*wtf is this*/ },
          ),
          "LayerElementUV", &[Data::I32(_/*idx of uv*/)] => |c| match_children!(
              self, c,
              "Version", &[Data::I32(_)] => |_| {},
              "Name", &[Data::String(_)] => |_| {},
              "MappingInformationType", &[Data::String(_)] => |c: usize| {
                  assert_eq!(self.kvs[c].values[0], Data::str("ByPolygonVertex"));
              },
              "ReferenceInformationType", &[Data::String(_)] => |c: usize| {
                  assert_eq!(self.kvs[c].values[0], Data::str("IndexToDirect"));
              },
              "UV", &[Data::F64Arr(_)] => |c: usize| {
                  let Data::F64Arr(ref arr) = &self.kvs[c].values[0] else {
                    todo!();
                  };
                  out.uv
                      .extend(arr.array_chunks::<2>().map(|uv| uv.map(|v| v as F)));
              },
              "UVIndex", &[Data::I32Arr(_)] => |c: usize| {
                  let Data::I32Arr(ref arr) = self.kvs[c].values[0] else {
                      todo!();
                  };
                  let idxs = arr
                      .iter()
                      .copied()
                      .inspect(|&idx| assert!(idx >= 0))
                      .map(|v| v as usize);
                  out.uv_idx.extend(idxs);
              }
          ),
          "LayerElementMaterial", &[Data::I32(_/*idx of mat*/)] => |c| {
              let mut mapping_kind = MappingKind::Uniform;
              match_children!(
                  self, c,
                  "Version", &[Data::I32(_)] => |_| {},
                  "Name", &[Data::String(_)] => |_| {},
                  "MappingInformationType", &[Data::String(_)] => |c: usize| {
                      mapping_kind = match self.kvs[c].values[0].as_str().unwrap() {
                          "AllSame" => MappingKind::Uniform,
                          "ByPolygon" => MappingKind::PerPolygon,
                          x => todo!("Unknown mapping kind {x:?}"),
                      };
                  },
                  "ReferenceInformationType", &[Data::String(_)] => |_| {},
                  "Materials", &[Data::I32(_) | Data::I32Arr(_)] => |c: usize| {
                      match &self.kvs[c].values[0] {
                          &Data::I32(i) => {
                              assert!(i >= 0);
                              assert_eq!(mapping_kind, MappingKind::Uniform);
                              out.global_mat = Some(i as usize);
                          }
                          Data::I32Arr(ref arr) => {
                              assert!(!arr.is_empty());
                              assert!(arr.iter().all(|&v| v >= 0));
                              if arr.len() == 1 {
                                  out.global_mat = Some(arr[0] as usize);
                                  return
                              }
                              assert_eq!(mapping_kind, MappingKind::PerPolygon);
                              let mat_idxs = arr.iter().map(|&i| i as usize);
                              out.per_face_mat.extend(mat_idxs);
                          }
                          x => todo!("{x:?}"),
                      }
                  },
              );
          },
          "LayerElementColor", &[] => |c| match_children!(
              self, c,
              "Version", &[Data::I32(_)] => |_| {},
              "Name", &[Data::String(_)] => |_| {},
              "MappingInformationType", &[Data::String(_)] => |c: usize| {
                  assert_eq!(self.kvs[c].values[0], Data::str("ByPolygonVertex"));
              },
              "ReferenceInformationType", &[Data::String(_)] => |_| {},
              "Colors", &[Data::F64Arr(_)] => |c: usize| {
                  let Some(v) = self.kvs[c].values[0].as_f64_arr() else {
                      todo!();
                  };
                  let vc = v.array_chunks::<3>().map(|v| v.map(|v| v as F));
                  out.vertex_colors.extend(vc);
              },
              "ColorIndex", &[Data::I32Arr(_)] => |c: usize| {
                  let Data::I32Arr(ref arr) = &self.kvs[c].values[0] else {
                      todo!();
                  };
                  let idxs = arr
                      .iter()
                      .copied()
                      .inspect(|&idx| assert!(idx >= 0))
                      .map(|v| v as usize);
                  out.vertex_color_idx.extend(idxs);
              },
          ),
          "Layer", &[Data::I32(0)] => |c| match_children!(
            self, c,
            "Version", &[Data::I32(_)] => |_| {},
            "Name", &[Data::String(_)] => |c| {},
            "MappingInformationType", &[Data::String(_)] => |_| {},
            "ReferenceInformationType", &[Data::String(_)] => |_| {},
            "LayerElement", &[] => |_| {},
          ),
          // omit for now
          "LayerElementSmoothing", &[] => |_| {},
          "LayerElementVisibility", &[] => |_| {},
        );
        out
    }

    pub fn to_scene(&self) -> FBXScene {
        let mut id_to_kv = HashMap::new();

        for (i, kv) in self.kvs.iter().enumerate() {
            let Some(p) = kv.parent else { continue };
            if let Some(id) = kv.id() {
                id_to_kv.insert(id, i);
            }
        }
        let mut fbx_scene = FBXScene::default();

        // parent->child pairs
        let mut connections = vec![];
        let mut prop_connections = vec![];

        let conns = self
            .find_root("Connections")
            .into_iter()
            .flat_map(|ci| &self.children[&ci]);
        for &child in conns {
            let kv = &self.kvs[child];
            assert_eq!(kv.key, "C");
            match kv.values.as_slice() {
                [oo, dst, src] if oo == &Data::str("OO") => {
                    let src = src.as_int().unwrap();
                    let dst = dst.as_int().unwrap();
                    println!(
                        "{src} {dst} {:?} {:?}",
                        self.kvs[id_to_kv[&src]].values, self.kvs[id_to_kv[&dst]].values
                    );
                    connections.push((src, dst));
                }
                [op, dst, src, name] if op == &Data::str("OP") => {
                    let src = src.as_int().unwrap();
                    let dst = dst.as_int().unwrap();
                    let name = name.as_str().unwrap();
                    prop_connections.push((src, dst, name));
                }
                x => todo!("{x:?}"),
            }

            assert!(!self.children.contains_key(&child));
        }

        let objects = self
            .find_root("Objects")
            .into_iter()
            .flat_map(|o| &self.children[&o]);
        for &o in objects {
            let kv = &self.kvs[o];
            assert_eq!(kv.values.len(), 3);
            let [id, name_objtype, classtag] = &kv.values[..] else {
                todo!("{:?}", kv.values);
            };
            let id = id.as_int().unwrap();
            let n_o = name_objtype.as_str().unwrap().split_once("\\x00\\x01");
            let Some((name, obj_type)) = n_o else {
                todo!("{name_objtype:?}");
            };

            let classtag = classtag.as_str().unwrap();

            match obj_type {
                "NodeAttribute" => match classtag {
                    "Light" => continue,
                    "Camera" => continue,
                    _ => todo!("NodeAttribute::{classtag} not handled"),
                },
                "Geometry" => match classtag {
                    "Mesh" => {
                        let mut fbx_mesh = self.parse_mesh(id, id_to_kv[&id]);
                        fbx_mesh.name = String::from(name);
                        fbx_scene.meshes.push(fbx_mesh);
                    }
                    _ => todo!("Geometry::{classtag} not handled"),
                },
                // Do not handle lights or cameras for now
                "Model" => match classtag {
                    "Mesh" => {
                        let kv = &self.kvs[id_to_kv[&id]];
                        let mut node = self.parse_node(id, id_to_kv[&id]);
                        node.name = String::from(name);
                        let parents = connections.iter().filter(|&&(_src, dst)| dst == id);
                        let mut num_parents = 0;
                        let new_idx = fbx_scene.nodes.len();
                        for &(parent_id, _) in parents {
                            if parent_id == 0 {
                                fbx_scene.root_nodes.push(new_idx);
                                num_parents += 1;
                                continue;
                            }
                            let parent = &self.kvs[id_to_kv[&parent_id]];
                            match parent.key.as_str() {
                                "CollectionExclusive" => continue,
                                x => todo!("{x:?} {}", kv.key),
                            }
                            num_parents += 1;
                        }
                        assert_eq!(num_parents, 1);

                        let conns = connections.iter().filter(|&&(src, _dst)| src == id);
                        for (_, c) in conns {
                            let c_kv = &self.kvs[id_to_kv[c]];
                            match c_kv.key.as_str() {
                                "Geometry" => {
                                    let Some(p) =
                                        fbx_scene.meshes.iter().position(|p| p.id == *c as usize)
                                    else {
                                        todo!("Load mesh lazily?");
                                    };
                                    node.mesh = Some(p);
                                }
                                // Don't handle materials yet
                                "Material" => continue,
                                x => todo!("{x:?}"),
                            }
                        }

                        fbx_scene.nodes.push(node);
                    }
                    "Light" => continue,
                    "Camera" => continue,
                    x => todo!("{x:?}"),
                },

                // Don't handle materials yet
                "Material" => continue,
                "Texture" => continue,

                "DisplayLayer" => continue,
                "Video" => continue,
                "Light" => continue,

                _ => todo!("{obj_type:?}"),
            }
        }

        root_fields!(
            self,
            "FBXHeaderExtension", &[],
            "FBXHeaderVersion", &[Data::I32(_)] => |v| {},
            "FBXVersion", &[Data::I32(_)] => |v| {},
            "EncryptionType", &[Data::I32(0)] => |v| {},
            "CreationTimeStamp", &[] => |v| {},
            "Creator", &[Data::String(_)] => |v| {},
            "SceneInfo", &[Data::String(_), Data::String(_)] => |v: usize| {
              match_children!(
                self, v,
                "Type", &[Data::String(_)] => |v| {},
                "Version", &[Data::I32(_)] => |v| {},
                "MetaData", &[] => |v| {},
                "Properties70", &[] => |v| {},
              );
            },
        );

        if let Some(file_id) = self.find_root("FileId") {
            let kv = &self.kvs[file_id];
            assert_matches!(kv.values.as_slice(), &[Data::Binary(_)]);
            let Data::Binary(ref b) = kv.values[0] else {
                unreachable!();
            };
            assert_eq!(b.len(), 16);
            fbx_scene.file_id.clone_from(b);
        }

        root_fields!(self, "CreationTime", &[Data::String(_)]);

        root_fields!(self, "Creator", &[Data::String(_)]);

        let settings = &mut fbx_scene.global_settings;

        root_fields!(
          self,
          "GlobalSettings", &[],
          "Version", &[Data::I32(_)] => |v| {},
          "Properties70", &[] => |v| {
            match_children!(
              self, v,
              "P",
              &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I32(_) | Data::I64(_) | Data::F64(_) | Data::String(_),
              ] |
              &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::F64(_), Data::F64(_), Data::F64(_)
              ]
              => |v: usize| {
                let vals = &self.kvs[v].values;
                let v4 = &vals[4];
                macro_rules! assign {
                  ($v: expr, $fn: ident) => {{
                    $v = *v4.$fn().unwrap_or(&$v);
                  }};
                }
                match vals[0].as_str().unwrap() {
                  "UpAxis" => assign!(settings.up_axis, as_i32),
                  "UpAxisSign" => assign!(settings.up_axis_sign, as_i32),
                  "FrontAxis" => assign!(settings.front_axis, as_i32),
                  "FrontAxisSign" => assign!(settings.front_axis_sign, as_i32),
                  "CoordAxis" => assign!(settings.coord_axis, as_i32),
                  "CoordAxisSign" => assign!(settings.coord_axis_sign, as_i32),
                  "OriginalUpAxis" => assign!(settings.og_up_axis, as_i32),
                  "OriginalUpAxisSign" => assign!(settings.og_up_axis_sign, as_i32),
                  "UnitScaleFactor" => assign!(settings.unit_scale_factor, as_f64),
                  "OriginalUnitScaleFactor" => assign!(settings.og_unit_scale_factor, as_f64),
                  // ignored
                  "AmbientColor" => return,
                  "DefaultCamera" => return,
                  "TimeMode" => return,
                  "TimeSpanStart" => return,
                  "TimeSpanStop" => return,
                  "CustomFrameRate" => return,
                  x => todo!("Unhandled Properties70 P {x:?}")
                };

              },
            );
          },
        );

        root_fields!(
          self,
          "Documents", &[],
          // I think this is only ever 1 for 1 scene
          "Count", &[Data::I32(1)] => |_| {},
          "Document", &[Data::I64(_), Data::String(_), Data::String(_)] => |v: usize| {
            assert_eq!(self.kvs[v].values[1].as_str().unwrap(), "Scene");
            assert_eq!(self.kvs[v].values[2].as_str().unwrap(), "Scene");
            match_children!(
              self, v,
              "Properties70", &[] => |v| {
                match_children!(self,v,

                "P",
                &[
                  Data::String(_), Data::String(_), Data::String(_),
                  Data::I32(_) | Data::String(_)
                ] |
                &[
                  Data::String(_), Data::String(_), Data::String(_),
                  Data::String(_), Data::String(_)
                ]
                => |v: usize| {
                    let vals = &self.kvs[v].values;
                    match vals[0].as_str().unwrap() {
                      "SourceObject" => {},
                      "ActiveAnimStackName" => {},
                      x => todo!("{x}"),
                    }
                  }
                );
              },

              "RootNode", &[Data::I64(_)] => |v| assert!(!self.children.contains_key(&v)),
            );
          },
        );

        // TODO what is this for?
        root_fields!(self, "References", &[]);

        root_fields!(
          self,
          "Definitions", &[],
          "Version", &[Data::I32(_)] => |v| {},
          "Count", &[Data::I32(_)] => |v| {},
          "ObjectType", &[Data::String(_)] => |v| match_children!(self, v,
            "Count", &[Data::I32(_)] => |v| {},
            "PropertyTemplate", &[Data::String(_)] => |v: usize| {
              assert_matches!(
                self.kvs[v].values[0].as_str().unwrap(),
                "FbxCamera" | "FbxMesh" | "FbxNode" | "FbxSurfacePhong",
              );
            }
          ),
        );

        // objects (handled earlier)
        // conections (handled earlier)

        root_fields!(
          self,
          "Takes", &[],
          "Current", &[Data::String(_)] => |v: usize| {},
        );

        fbx_scene
    }
}

pub fn parse_tokens(mut tokens: impl Iterator<Item = Token>) -> KVs {
    let mut kvs = KVs::default();
    kvs.parse_scope(&mut tokens, None);
    kvs
}

pub fn tokenize_binary(mut src: impl BufRead + Seek) -> io::Result<Vec<Token>> {
    let mut buf = [0u8; MAGIC_LEN];
    src.read_exact(&mut buf)?;
    assert_eq!(&buf, MAGIC, "FBX Header mismatch");

    let mut version = [0u8; 4];
    src.read_exact(&mut version)?;
    let version = u32::from_le_bytes(version);

    let is_64_bit = version >= 7500;

    let mut output_tokens = vec![];
    // https://github.com/assimp/assimp/blob/53d4663f298ffa629505072fc01a5219c2b42b3e/code/AssetLib/FBX/FBXBinaryTokenizer.cpp#L451
    let mut curr_read = MAGIC_LEN + size_of::<u32>();
    loop {
        let (cont, read) = read_scope(&mut src, is_64_bit, &mut output_tokens, curr_read)?;
        if !cont {
            break;
        }
        curr_read += read;
    }

    Ok(output_tokens)
}

fn read_scope(
    src: &mut (impl BufRead + Seek),
    is_64_bit: bool,
    output_tokens: &mut Vec<Token>,
    prev_read: usize,
) -> io::Result<(bool, usize)> {
    let mut read = 0;
    macro_rules! read_buf {
        ($len: expr) => {{
            let mut buf = vec![0u8; $len];
            src.read_exact(&mut buf)?;
            read += $len;
            buf
        }};
    }
    macro_rules! read_word {
        (bool) => {{
            read_word!(u8) == 1
        }};
        ($t: ty) => {{
            let mut v = [0u8; size_of::<$t>()];
            // here is failing
            src.read_exact(&mut v).expect("tmp");
            read += size_of::<$t>();
            <$t>::from_le_bytes(v)
        }};
        ($len: expr) => {{
            let mut buf = vec![0u8; $len];
            src.read_exact(&mut buf)?;
            read += $len;
            buf
        }};
        () => {{
            if is_64_bit {
                read_word!(u64)
            } else {
                read_word!(u32) as u64
            }
        }};
    }

    macro_rules! read_string {
        ($is_long: expr, $allow_null: expr) => {{
            let len = if $is_long {
                read_word!(u32)
            } else {
                read_word!(u8) as u32
            };
            let mut buf = vec![0u8; len as usize];
            src.read_exact(&mut buf)?;
            read += buf.len();
            assert!($allow_null || buf.iter().all(|&v| v != b'\0'));
            buf.escape_ascii().to_string()
        }};
    }

    macro_rules! read_array {
        (bool) => {{
            let len = read_word!(u32) as usize;
            let enc = read_word!(u32) as usize;
            let comp_len = read_word!(u32) as usize;

            let stride = size_of::<bool>();
            assert_eq!(
                len * stride,
                comp_len,
                "Mismatch in read size: {len} * {stride} != {comp_len}"
            );
            let mut out = Vec::with_capacity(len);
            match enc {
                0 => {
                    for _ in 0..len {
                        out.push(read_word!(bool));
                    }
                }
                1 => todo!("zip/deflate encoding"),
                _ => todo!("wtf"),
            }
            out
        }};
        ($ty: ty) => {{
            let len = read_word!(u32) as usize;
            let enc = read_word!(u32) as usize;
            let comp_len = read_word!(u32) as usize;

            let stride = size_of::<$ty>();
            let mut out = Vec::with_capacity(len);
            match enc {
                0 => {
                    assert_eq!(len * stride, comp_len);
                    for _ in 0..len {
                        out.push(read_word!($ty));
                    }
                }
                1 => {
                    let data = read_buf!(comp_len);
                    let mut decoder = zune_inflate::DeflateDecoder::new(&data);
                    let deflated = decoder.decode_zlib().unwrap();
                    assert_eq!(deflated.len(), len * stride);
                    let elems = deflated
                        .into_iter()
                        .array_chunks::<{ size_of::<$ty>() }>()
                        .map(<$ty>::from_le_bytes);
                    out.extend(elems);
                }
                e => todo!("wtf encoding {e}"),
            }
            out
        }};
    }

    let end_offset = read_word!();

    // this marks the end of the tokens of the tokens of the tokens of the tokens
    if end_offset == 0 {
        return Ok((false, read));
    }
    let block_len = end_offset - (prev_read as u64);

    let prop_count = read_word!();
    let prop_len = read_word!();
    let scope_name = read_string!(false, false);
    assert_ne!(scope_name, "");

    output_tokens.push(Token::Key(scope_name.clone()));

    let curr_read = read;
    for _pi in 0..prop_count {
        let Some(d) = read_word!(u8).as_ascii() else {
            todo!();
        };
        let data = match d {
            // TODO are these signed or unsigned?
            Char::CapitalY => Data::I16(read_word!(i16)),

            Char::CapitalI => Data::I32(read_word!(i32)),
            Char::SmallI => Data::I32Arr(read_array!(i32)),

            Char::CapitalL => Data::I64(read_word!(i64)),
            Char::SmallL => Data::I64Arr(read_array!(i64)),

            Char::CapitalF => Data::F32(read_word!(f32)),
            Char::SmallF => Data::F32Arr(read_array!(f32)),

            Char::CapitalD => Data::F64(read_word!(f64)),
            Char::SmallD => Data::F64Arr(read_array!(f64)),

            Char::CapitalR => {
                let len = read_word!(u32);
                Data::Binary(read_buf!(len as usize))
            }

            Char::SmallC => Data::BoolArr(read_array!(bool)),

            Char::SmallB => {
                // TODO not sure what this is, but skip it for now
                let _ = read_word!(u32);
                let _ = read_word!(u32);
                let len = read_word!(u32);
                src.seek(SeekFrom::Current(len as i64));
                read += len as usize;
                Data::Unknown(len as usize)
            }

            Char::CapitalS => Data::String(read_string!(true, true)),
            Char::CapitalC => Data::Bool(read_word!(bool)),

            c => todo!("unhandled {c:?} (u8 = {})", c.to_u8()),
        };
        output_tokens.push(Token::Data(data));
    }
    assert_eq!((read - curr_read) as u64, prop_len, "{scope_name}");

    let sentinel_block_len = if is_64_bit {
        size_of::<u64>() * 3 + 1
    } else {
        size_of::<u32>() * 3 + 1
    } as u64;

    if (read as u64) < block_len {
        assert!(block_len - read as u64 >= sentinel_block_len);

        output_tokens.push(Token::ScopeStart);
        while (read as u64) + sentinel_block_len < block_len {
            read += read_scope(src, is_64_bit, output_tokens, prev_read + read)?.1;
        }
        if output_tokens.last() == Some(&Token::ScopeStart) {
            output_tokens.pop();
        } else {
            output_tokens.push(Token::ScopeEnd);
        }

        let sentinel = read_word!(sentinel_block_len as usize);
        assert!(sentinel.iter().all(|&v| v == b'\0'));
    }

    assert_eq!(read as u64, block_len);

    Ok((true, read))
}

pub fn load<P: AsRef<Path>>(p: P) -> std::io::Result<FBXScene> {
    use std::fs::File;
    use std::io::BufReader;
    let f = File::open(p)?;
    let tokens = tokenize_binary(BufReader::new(f)).expect("Failed to tokenize FBX");
    let kvs = parse_tokens(tokens.into_iter());
    Ok(kvs.to_scene())
}

#[test]
fn test_parse_fbx() {
    use std::fs::File;
    use std::io::BufReader;
    let f = File::open("src/fbx/test_data/cube.fbx").unwrap();
    let tokens = tokenize_binary(BufReader::new(f)).expect("Failed to tokenize FBX");
    let kvs = parse_tokens(tokens.into_iter());

    let scene = kvs.to_scene();
}
