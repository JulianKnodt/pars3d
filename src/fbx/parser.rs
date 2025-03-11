use super::{
    FBXAnimCurve, FBXAnimCurveNode, FBXAnimLayer, FBXAnimStack, FBXBlendshape,
    FBXBlendshapeChannel, FBXCluster, FBXMaterial, FBXMesh, FBXMeshMaterial, FBXNode, FBXScene,
    FBXSkin, FBXTexture, RefKind, VertexMappingKind,
};
use crate::{FaceKind, F};

use std::array::from_fn;
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

const STRICT: bool = cfg!(feature = "strict_fbx");

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
    cast!(as_f32_arr, &[f32], F32Arr);
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
    /*
    fn as_usize(&self) -> Option<usize> {
        assert!(self.as_int()?
        match *self {
            Data::I64(v) => Some(v as usize),
            Data::I32(v) => Some(v as usize),
            _ => None,
        }
    }
    */
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

/// One token of a tokenized FBX file.
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

macro_rules! todo_if_strict {
  ($( $x: tt )*) => {
    if STRICT {
      eprintln!("Found unexpected case when parsing FBX file");
      eprintln!("Please file a bug to\n\nhttps://github.com/JulianKnodt/pars3d/issues/new?template=Blank+issue");
      eprintln!("I'd appreciate if you also attached the mesh as well.");
      todo!($($x)*);
    }
  }
}

macro_rules! match_children {
  ($self: ident, $i: expr $(, $key:expr, $mtch: pat => $val:expr )* $(,)? ) => {{
    let kv = &$self.kvs[$i];
    /* Debug
    #[allow(unused)]
    let mut matches: [(&'static str, bool); _] = [$(($key, false),)*];
    */

    for &c in $self.children.get(&$i).map(Vec::as_slice).unwrap_or(&[]) {
      let c_kv = &$self.kvs[c];

      match c_kv.key.as_str() {
        $($key => {
          assert_matches!(c_kv.values.as_slice(), $mtch, "{} {}", $key, kv.key);
          //matches.iter_mut().find(|v| v.0 == $key).unwrap().1 = true;
          $val(c)
        })*
        x => todo_if_strict!("Unhandled key {x:?} in {} children", kv.key),
      }
    }

    /*
    for (k, did_match) in matches {
      if !did_match {
        println!("Could not find {k} in {} ({}:{})", kv.key, file!(), line!());
      }
    }
    */
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
          x => todo_if_strict!("Unhandled {x} in {}", $key),
        }
      }
    }
  }}
}

/// Represents the set of key-value pairs that make up an FBX file.
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

    fn parse_node(&self, out: &mut FBXNode, node_id: i64, kvi: usize) {
        assert!(node_id >= 0);
        out.id = node_id as usize;

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "Properties70", &[] => |c| match_children!(
            self, c,
            "P", &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::F64(_), Data::F64(_), Data::F64(_)
            ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I32(_) | Data::String(_),
            ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I16(_), Data::I16(_), Data::I16(_)
            ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I32(_), Data::I32(_), Data::I32(_)
            ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_), Data::F64(_)
            ] => |c: usize| {
                let vals = &self.kvs[c].values;
                match vals[0].as_str().unwrap() {
                  "Lcl Rotation" => {
                    out.transform.rotation = [4,5,6].map(|i| vals[i].as_float().unwrap());
                  },
                  "Lcl Scaling" => {
                    out.transform.scale = [4,5,6].map(|i| vals[i].as_float().unwrap());
                  },
                  "Lcl Translation" => {
                    out.transform.translation = [4,5,6].map(|i| vals[i].as_float().unwrap());
                  },
                  "PreRotation" => {},

                  "DefaultAttributeIndex" => {},
                  "InheritType" => {},

                  /* Not sure how to handle these */
                  "RotationPivot" => {},
                  "ScalingPivot" => {},
                  "RotationOrder" => {},
                  "RotationActive" => {},
                  "ScalingMax" => {},

                  "currentUVSet" => {},
                  "filmboxTypeID" => {},
                  "Visibility" => out.hidden = vals[4].as_float() == Some(0.0),

                  "UDP3DSMAX" => {},
                  "MaxHandle" => {},

                  x if x.starts_with("mr") => {},

                  x => todo_if_strict!("Unhandled node key: {x:?}"),
                }
             },
          ),
          "MultiLayer", &[Data::I32(_)] => |_| {},
          "Culling", &[Data::String(_)] => |c: usize| {
            assert_matches!(self.kvs[c].values[0].as_str(), Some("CullingOff" | "CullingOn"));
          },
          "MultiTake", &[Data::I32(_)] => |c: usize| {
            assert_matches!(self.kvs[c].values[0].as_int(), Some(0));
          },
          "Shading", &[Data::Bool(_)] => |_| {},
        );
    }
    fn parse_null(&self, null_id: i64, kvi: usize) {
        assert!(null_id >= 0);

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "Properties70", &[] => |c| match_children!(
            self, c,
            "P", &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_), Data::F64(_), Data::F64(_),
            ] | &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::I32(_) | Data::F64(_)
            ] => |c: usize| {
              let vals = &self.kvs[c].values;
              match vals[0].as_str().unwrap() {
                "Look" => {},
                "Color" => {},
                "Size" => {},
                x => todo_if_strict!("Unhandled property on Null Node {x:?}"),
              }
            }
          ),
          "TypeFlags", &[Data::String(_)] => |c: usize| {
              assert_matches!(self.kvs[c].values[0].as_str(), Some("Null"));
          },
        );
    }
    fn parse_limb_node(&self, limb_node_id: i64, kvi: usize) {
        assert!(limb_node_id >= 0);

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "TypeFlags", &[Data::String(_)] => |c: usize| {
            assert_matches!(self.kvs[c].values[0].as_str(), Some("Skeleton"));
            match_children!(self, c);
          },
          "Properties70", &[] => |c| match_children!(
            self, c, "P", &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_),
            ] => |c: usize| {
              let vals = &self.kvs[c].values;
              match vals[0].as_str().unwrap() {
                "Size" => {},
                x => todo_if_strict!("Unhandled property on Limb Node {x:?}"),
              }
            }
          ),
        );
    }
    fn parse_blendshape(&self, out: &mut FBXBlendshape, id: i64, kvi: usize) {
        assert!(id >= 0);

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "Indexes", &[Data::I32Arr(_)] => |c: usize| {
            let idxs = self.kvs[c].values[0].as_i32_arr().unwrap();
            out.indices.extend(idxs.iter().map(|&v| v as usize));
          },
          "Vertices", &[Data::F64Arr(_)] => |c: usize| {
              let v_arr: &[f64] = self.kvs[c].values[0].as_f64_arr().unwrap();
              let v = v_arr
                  .iter()
                  .array_chunks::<3>()
                  .map(|[a, b, c]| [*a as F, *b as F, *c as F]);
              out.v.clear();
              out.v.extend(v);
              match_children!(self, c);
          },
          "Normals", &[Data::F64Arr(_)] => |c: usize| {
              let v_arr: &[f64] = self.kvs[c].values[0].as_f64_arr().unwrap();
              let v = v_arr
                  .iter()
                  .array_chunks::<3>()
                  .map(|[a, b, c]| [*a as F, *b as F, *c as F]);
              out.n.clear();
              out.n.extend(v);
              match_children!(self, c);
          },
        );
    }
    fn parse_blendshape_channel(&self, bs_chan: &mut FBXBlendshapeChannel, id: i64, kvi: usize) {
        assert!(id >= 0);

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "DeformPercent", &[Data::F64(_)] => |c: usize| {
            bs_chan.deform_percent = *self.kvs[c].values[0].as_f64().unwrap() as F;
          },
          "FullWeights", &[Data::F64Arr(_)] => |_| {},
          "Properties70", &[] => |c| match_children!(
            self, c,
            "P", &[
              Data::String(_), Data::String(_), Data::String(_),
              Data::String(_), Data::F64(_)
            ] => |c: usize| {
              let vals = &self.kvs[c].values;
              match vals[0].as_str().unwrap() {
                "DeformPercent" => {},
                x => todo_if_strict!("{x}"),
              }
            },
          ),
        );
    }
    fn parse_skin(&self, _fbx_skin: &mut FBXSkin, skin_id: i64, kvi: usize) {
        assert!(skin_id >= 0);
        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          // Damn you FBX
          "Link_DeformAcuracy", &[Data::F64(_)] => |c| match_children!(self, c),
          "SkinningType", &[Data::String(_)] => |c: usize| {
            let ty = self.kvs[c].values[0].as_str().unwrap();
            assert_matches!(ty, "Linear");
          },
        );
    }
    fn parse_anim_stack(&self, anim_stack: &mut FBXAnimStack, anim_stack_id: i64, kvi: usize) {
        assert!(anim_stack_id >= 0);
        match_children!(
          self, kvi,
          "Properties70", &[] => |c| match_children!(
            self, c, "P",
            &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::I64(_)
            ] => |c: usize| {
              let vals = &self.kvs[c].values;
              match vals[0].as_str().unwrap() {
                "LocalStart" => {
                  anim_stack.local_start = *vals[4].as_i64().unwrap();
                },
                "LocalStop" => {
                  anim_stack.local_stop = *vals[4].as_i64().unwrap();
                },
                "ReferenceStart" => {
                  anim_stack.ref_start = *vals[4].as_i64().unwrap();
                },
                "ReferenceStop" => {
                  anim_stack.ref_stop = *vals[4].as_i64().unwrap();
                },
                x => todo_if_strict!("{x:?}"),
              }
            },
          ),
        );
    }
    fn parse_pose(&self, fbx_scene: &mut FBXScene, id: i64, kvi: usize) {
        assert!(id >= 0);
        assert_eq!(self.kvs[kvi].key, "Pose");
        let pose_idx = fbx_scene.pose_by_id_or_new(id as usize);
        match_children!(
          self, kvi,
          "Type", &[Data::String(_)] => |c: usize| {
            let v = self.kvs[c].values[0].as_str().unwrap();
            assert_eq!(v, "BindPose");
          },
          "Version", &[Data::I32(_)] => |_| {},
          // how many nodes are associated with this pose?
          "NbPoseNodes", &[Data::I32(_)] => |_| {},
          "PoseNode", &[] => |c| match_children!(
            self, c,
            "Node", &[Data::I64(_)] => |c: usize| {
              let node_id = *self.kvs[c].values[0].as_i64().unwrap();
              assert!(node_id >= 0);
              let node_idx = fbx_scene.node_by_id_or_new(node_id as usize);
              fbx_scene.poses[pose_idx].nodes.push(node_idx);
            },
            "Matrix", &[Data::F64Arr(_)] => |c: usize| {
              let mat = self.kvs[c].values[0].as_f64_arr().unwrap();
              assert_eq!(mat.len(), 16);
              fbx_scene.poses[pose_idx].matrices.push(from_fn(|i| from_fn(|j| mat[i * 4 + j] as F)));
            }
          ),
        );
    }
    fn parse_anim_layer(&self, _anim_layer: &mut FBXAnimLayer, anim_layer_id: i64, kvi: usize) {
        assert!(anim_layer_id >= 0);
        assert_eq!(self.kvs[kvi].key, "AnimationLayer");
        match_children!(self, kvi);
    }
    fn parse_anim_curve(&self, anim: &mut FBXAnimCurve, anim_curve_id: i64, kvi: usize) {
        assert!(anim_curve_id >= 0);
        assert_eq!(self.kvs[kvi].key, "AnimationCurve");
        match_children!(
          self, kvi,
          "Default", &[Data::F64(_)] => |c: usize| {
            anim.default = *self.kvs[c].values[0].as_f64().unwrap() as F;
          },
          "KeyVer", &[Data::I32(_)] => |_| {},
          "KeyTime", &[Data::I64Arr(_)] => |c: usize| {
            let val = self.kvs[c].values[0].as_i64_arr().unwrap();
            anim.times.extend(val.iter().map(|&v| v as u32));
          },
          "KeyValueFloat", &[Data::F32Arr(_)] => |c: usize| {
            let val = self.kvs[c].values[0].as_f32_arr().unwrap();
            anim.values.extend(val.iter().map(|&v| v as F));
          },
          "KeyAttrFlags", &[Data::I32Arr(_)] => |c: usize| {
            let val = self.kvs[c].values[0].as_i32_arr().unwrap();
            anim.flags.extend(val.iter().copied());
          },
          "KeyAttrDataFloat", &[Data::F32Arr(_)] => |c: usize| {
            let val = self.kvs[c].values[0].as_f32_arr().unwrap();
            anim.data.extend(val.iter().map(|&v| v as F));
          },
          "KeyAttrRefCount", &[Data::I32Arr(_)] => |c: usize| {
            let val = self.kvs[c].values[0].as_i32_arr().unwrap();
            assert!(val.iter().all(|&v| v >= 0));
            anim.ref_count.extend(val.iter().copied());
          },
        );
    }
    fn parse_anim_curve_node(
        &self,
        _anim_curve_node: &mut FBXAnimCurveNode,
        anim_curve_node_id: i64,
        kvi: usize,
    ) {
        assert!(anim_curve_node_id >= 0);
        assert_eq!(self.kvs[kvi].key, "AnimationCurveNode");
        match_children!(
          self, kvi,
          "Properties70", &[] => |c| match_children!(
            self, c, "P", &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::I16(_) | Data::I32(_) | Data::F64(_),
            ] => |c: usize| {
              let val = &self.kvs[c].values;
              match val[0].as_str().unwrap() {
                "d|filmboxTypeID" => {},
                "d|lockInfluenceWeights" => {},
                "d|X" => {},
                "d|Y" => {},
                "d|Z" => {},
                "d|DeformPercent" => {},
                x => todo_if_strict!("Unknown anim curve node P70 {x:?}"),
              }
            },
          ),
        );
    }
    fn parse_cluster(&self, fbx_cl: &mut FBXCluster, cluster_id: i64, kvi: usize) {
        assert!(cluster_id >= 0);

        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "UserData", &[Data::String(_), Data::String(_)] => |c| {
            match_children!(self, c);
          },
          // Damn you FBX
          "Indexes", &[Data::I32Arr(_)] => |c: usize| {
            let idxs = self.kvs[c].values[0].as_i32_arr().unwrap();
            fbx_cl.indices.extend(idxs.iter().map(|&v| TryInto::<usize>::try_into(v).unwrap()));
          },
          "Weights", &[Data::F64Arr(_)] => |c: usize| {
            let ws = self.kvs[c].values[0].as_f64_arr().unwrap();
            fbx_cl.weights.extend(ws.iter().map(|&v| v as F));
          },
          "Transform", &[Data::F64Arr(_)] => |c: usize| {
            let tform = self.kvs[c].values[0].as_f64_arr().unwrap();
            assert_eq!(tform.len(), 16);
            fbx_cl.tform = from_fn(|i| from_fn(|j| tform[i * 4 + j] as F));
          },
          "TransformLink", &[Data::F64Arr(_)] => |c: usize| {
            let tform_link = self.kvs[c].values[0].as_f64_arr().unwrap();
            assert_eq!(tform_link.len(), 16);
            fbx_cl.tform_link = from_fn(|i| from_fn(|j| tform_link[i * 4 + j] as F));
          },
          // not sure what this is for
          "TransformAssociateModel", &[Data::F64Arr(_)] => |c: usize| {
            let tam = self.kvs[c].values[0].as_f64_arr().unwrap();
            assert_eq!(tam.len(), 16);
            fbx_cl.tform_assoc_model = from_fn(|i| from_fn(|j| tam[i * 4 + j] as F));
          },
        );
    }
    fn parse_material(&self, out: &mut FBXMaterial, mat_id: i64, kvi: usize) {
        assert!(mat_id >= 0);
        out.id = mat_id as usize;
        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "ShadingModel", &[Data::String(_)] => |c: usize| {
            let shading_model = self.kvs[c].values[0].as_str().unwrap();
            assert_matches!(shading_model, "phong" | "Phong" | "lambert" | "unknown");
          },
          "Properties70", &[] => |c| match_children!(
            self, c,
            "P", &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_), Data::F64(_), Data::F64(_),
            ] | [
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_) | Data::String(_) | Data::I32(_) | Data::F32(_)
            ] | [
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_), Data::F64(_)
            ] | [
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
            ]=> |c: usize| {
              let kv = &self.kvs[c];
              match kv.values[0].as_str().unwrap() {
                "Diffuse" | "DiffuseColor" => out.diffuse_color = [4,5,6].map(
                  |i| *kv.values[i].as_f64().unwrap() as F
                ),
                "DiffuseFactor" => {},

                "Ambient" | "AmbientColor" => {},
                "AmbientFactor" => {},
                "BumpFactor" => {},
                "SpecularColor" => out.specular_color = [4,5,6].map(
                  |i| *kv.values[i].as_f64().unwrap() as F
                ),
                "SpecularFactor" => {},
                "Specular" => {},

                "Shininess" => {},
                "ShininessExponent" => {},
                "ShininessFactor" => {},
                "Reflectivity" => {},
                "ReflectionColor" => {},
                "ReflectionFactor" => {},
                "Opacity" => {},
                "TransparentColor" => {},
                "TransparencyFactor" => {},
                "Emissive" => {},
                "EmissiveColor" => {},
                "EmissiveFactor" => {},

                "ShadingModel" => {},

                // monopoly powers
                x if x.starts_with("Maya") => {},

                x => todo_if_strict!("Unknown material property {x:?}"),
              }
            },
          ),
          "MultiLayer", &[Data::I32(_)] => |c| match_children!(self, c),
        );
    }
    fn parse_texture(&self, out: &mut FBXTexture, tex_id: i64, kvi: usize) {
        assert!(tex_id >= 0);
        out.id = tex_id as usize;
        match_children!(
          self, kvi,
          "Version", &[Data::I32(_)] => |_| {},
          "Type", &[Data::String(_)] => |_| {},
          "TextureName", &[Data::String(_)] => |_| {},
          "Properties70", &[] => |c: usize| match_children!(
            self, c, "P", &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::I32(_) | Data::String(_)
            ] | &[
              Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              Data::F64(_), Data::F64(_), Data::F64(_)
            ] => |c: usize| {
              let vals = &self.kvs[c].values;
              match vals[0].as_str().unwrap() {
                "CurrentTextureBlendMode" => {},
                "UVSet" => {},
                "UseMaterial" => {},
                "Translation" => {},
                "Rotation" => {},
                "WrapModeU" => {},
                "WrapModeV" => {},
                x => todo_if_strict!("Unknown texture property {x:?}"),
              }
            }
          ),
          "FileName", &[Data::String(_)] => |c: usize| {
            out.file_name = String::from(self.kvs[c].values[0].as_str().unwrap());
          },
          "RelativeFilename", &[Data::String(_)] => |_| {},
          "Media", &[Data::String(_)] => |_| {},
          "ModelUVTranslation", &[Data::F64(_), Data::F64(_)] => |_| {},
          "ModelUVScaling", &[Data::F64(_), Data::F64(_)] => |_| {},
          "Texture_Alpha_Source", &[Data::String(_)] => |_| {},
          "Cropping", &[Data::I32(_), Data::I32(_), Data::I32(_), Data::I32(_)] => |_| {},
        );
    }

    fn parse_mesh(&self, out: &mut FBXMesh, mesh_id: i64, kvi: usize) {
        assert!(mesh_id >= 0);
        assert_eq!(out.id, mesh_id as usize);

        match_children!(
          self, kvi,
          "Properties70", &[] => |c| match_children!(self, c, "P",
              &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::F64(_), Data::F64(_), Data::F64(_)
              ] => |c:usize| {
                let vals = &self.kvs[c].values;
                match vals[0].as_str().unwrap() {
                  "Color" => {},
                  x => todo_if_strict!("Unknown mesh property {x:?}"),
                }
              },
          ),
          "Vertices", &[Data::F64Arr(_)] => |c: usize| {
              let v_arr: &[f64] = self.kvs[c].values[0].as_f64_arr().unwrap();
              assert_eq!(v_arr.len() % 3, 0);
              let v = v_arr
                  .iter()
                  .array_chunks::<3>()
                  .map(|[a, b, c]| [*a as F, *b as F, *c as F]);
              out.v.extend(v);
              match_children!(self, c);
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
                      assert!(curr_face.len() >= 3);
                      let f = std::mem::replace(&mut curr_face, FaceKind::empty());
                      out.f.push(f);
                  }
              }
              assert!(curr_face.is_empty());
          },
          "Smoothness", &[Data::I32(_)] => |c: usize| {
            let _smoothness = self.kvs[c].values[0].as_i32().unwrap();
          },
          "Edges", &[Data::I32Arr(_)] => |_| { /* No idea what to do here */ },
          "LayerElementNormal", &[Data::I32(_/*index of normal*/)] => |c| {
              match_children!(
                self, c,
                "Version", &[Data::I32(_)] => |_| {},
                "Name", &[Data::String(_)] => |_| {},
                "MappingInformationType", &[Data::String(_)] => |c: usize| {
                    let map_str = self.kvs[c].values[0].as_str().unwrap();
                    out.n.map_kind = VertexMappingKind::from_str(map_str);
                },
                "ReferenceInformationType", &[Data::String(_)] => |c: usize| {
                    let ref_str = self.kvs[c].values[0].as_str().unwrap();
                    out.n.ref_kind = RefKind::from_str(ref_str);
                },
                "Normals", &[Data::F64Arr(_)] => |c: usize| {
                    let gc = &self.kvs[c];
                    let Data::F64Arr(ref ns) = &gc.values[0] else {
                      unreachable!();
                    };
                    assert_eq!(ns.len() % 3, 0);
                    let iter = ns.array_chunks::<3>().map(|n| n.map(|v| v as F));
                    out.n.values.extend(iter);
                },
                "NormalsIndex", &[Data::I32Arr(_)] => |c: usize| {
                    let gc = &self.kvs[c];
                    let Data::I32Arr(ref arr) = &gc.values[0] else {
                      unreachable!();
                    };
                    let idxs = arr
                        .iter()
                        .copied()
                        .inspect(|&idx| assert!(idx >= 0))
                        .map(|v| v as usize);
                    out.n.indices.extend(idxs);
                },
                "NormalsW", &[Data::F64Arr(_)] => |_| { /*normals winding, not sure*/ },
            );
          },
          "LayerElementUV", &[Data::I32(_/*idx of uv*/)] => |c| match_children!(
              self, c,
              "Version", &[Data::I32(_)] => |_| {},
              "Name", &[Data::String(_)] => |_| {},
              "MappingInformationType", &[Data::String(_)] => |c: usize| {
                  let map_str = self.kvs[c].values[0].as_str().unwrap();
                  out.uv.map_kind = VertexMappingKind::from_str(map_str);
              },
              "ReferenceInformationType", &[Data::String(_)] => |c: usize| {
                  let ref_str = self.kvs[c].values[0].as_str().unwrap();
                  out.uv.ref_kind = RefKind::from_str(ref_str);
              },
              "UV", &[Data::F64Arr(_)] => |c: usize| {
                  let Data::F64Arr(ref arr) = &self.kvs[c].values[0] else {
                    unreachable!();
                  };
                  assert_eq!(arr.len() % 2, 0);
                  let uvs = arr.array_chunks::<2>().map(|uv| uv.map(|v| v as F));
                  out.uv.values.extend(uvs);
              },
              "UVIndex", &[Data::I32Arr(_)] => |c: usize| {
                  let Data::I32Arr(ref arr) = self.kvs[c].values[0] else {
                      unreachable!();
                  };
                  let idxs = arr
                      .iter()
                      .copied()
                      .inspect(|&idx| assert!(idx >= 0, "Found negative UV index {idx:?}"))
                      .map(|v| v as usize);
                  out.uv.indices.extend(idxs);
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
                              out.mat = FBXMeshMaterial::Global(i as usize);
                          }
                          Data::I32Arr(ref arr) => {
                              assert!(arr.iter().all(|&v| v >= 0));
                              if arr.is_empty() {
                                  out.mat = FBXMeshMaterial::None;
                                  return;
                              } else if arr.len() == 1 {
                                  out.mat = FBXMeshMaterial::Global(arr[0] as usize);
                                  return;
                              }
                              let first = arr[0];
                              let all_same = arr[1..].iter().all(|&v| v == first);
                              if all_same {
                                  out.mat = FBXMeshMaterial::Global(arr[0] as usize);
                                  return;
                              }
                              assert_eq!(mapping_kind, MappingKind::PerPolygon);
                              let mat_idxs = arr.iter().map(|&i| i as usize).collect::<Vec<_>>();
                              out.mat = FBXMeshMaterial::PerFace(mat_idxs);
                          }
                          x => todo!("Unknown material kind {x:?}"),
                      }
                  },
              );
          },
          "LayerElementColor", &[] | &[Data::I32(_)] => |c| match_children!(
              self, c,
              "Version", &[Data::I32(_)] => |_| {},
              "Name", &[Data::String(_)] => |_| {},
              "MappingInformationType", &[Data::String(_)] => |c: usize| {
                  assert_eq!(self.kvs[c].values[0], Data::str("ByPolygonVertex"));
              },
              "ReferenceInformationType", &[Data::String(_)] => |c: usize| {
                  assert_matches!(
                    self.kvs[c].values[0].as_str().unwrap(),
                    "Direct" | "IndexToDirect"
                  );
              },
              "Colors", &[Data::F64Arr(_)] => |c: usize| {
                  let Some(v) = self.kvs[c].values[0].as_f64_arr() else {
                      unreachable!();
                  };
                  let vc = v.array_chunks::<3>().map(|v| v.map(|v| v as F));
                  out.color.values.extend(vc);
              },
              "ColorIndex", &[Data::I32Arr(_)] => |c: usize| {
                  let Data::I32Arr(ref arr) = &self.kvs[c].values[0] else {
                      unreachable!();
                  };
                  let idxs = arr
                      .iter()
                      .copied()
                      .inspect(|&idx| assert!(idx >= 0))
                      .map(|v| v as usize);
                  out.color.indices.extend(idxs);
              },
          ),
          "Layer", &[Data::I32(_)] => |c| match_children!(
            self, c,
            "Version", &[Data::I32(_)] => |_| {},
            "LayerElement", &[] => |c| match_children!(
              self, c,
              "Type", &[Data::String(_)] => |c: usize| {
                assert_matches!(
                  self.kvs[c].values[0].as_str().unwrap(),
                  "LayerElementNormal" | "LayerElementUV" | "LayerElementMaterial" |
                  "LayerElementBinormal" | "LayerElementTangent" | "LayerElementSmoothing" |
                  "LayerElementColor" | "LayerElementVisibility" | "LayerElementUserData" |
                  "LayerElementPolygonGroup"
                );
              },
              "TypedIndex", &[Data::I32(_)] => |_| {},
            ),
          ),
          // omit for now
          "LayerElementSmoothing", &[] | &[Data::I32(_)] => |_| {},
          "LayerElementVisibility", &[] | &[Data::I32(_)] => |_| {},
          "LayerElementBinormal", &[Data::I32(_/*idx*/)] => |_| {},
          "LayerElementTangent", &[Data::I32(_/*idx*/)] => |_| {},

          "LayerElementUserData", &[Data::I32(_/*idx*/)] => |_| {},

          "PreviewDivisionLevels", &[Data::I32(_)] => |_| {},
          "RenderDivisionLevels", &[Data::I32(_)] => |_| {},
          "DisplaySubdivisions", &[Data::I32(_)] => |_| {},
          "BoundaryRule", &[Data::I32(_)] => |_| {},
          "PreserveBorders", &[Data::I32(_)] => |_| {},
          "PreserveHardEdges", &[Data::I32(_)] => |_| {},
          "PropagateEdgeHardness", &[Data::I32(_)] => |_| {},
        );
    }

    pub fn to_scene(&self) -> FBXScene {
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
                    connections.push((src, dst));
                }
                [op, dst, src, name] if op == &Data::str("OP") => {
                    let src = src.as_int().unwrap();
                    let dst = dst.as_int().unwrap();
                    let name = name.as_str().unwrap();
                    prop_connections.push((src, dst, name));
                }
                x => todo!("Unknown connection in FBX {x:?}"),
            }

            assert!(!self.children.contains_key(&child));
        }

        // connections by source id or destination id
        macro_rules! conns {
            ($src: expr =>) => {{
                connections
                    .iter()
                    .filter(|&&(src, _dst)| src == $src)
                    .map(|v| v.1)
            }};
            (=> $dst: expr) => {{
                connections
                    .iter()
                    .filter(|&&(_src, dst)| dst == $dst)
                    .map(|v| v.0)
            }};
            (PROP $src: expr =>) => {{
                prop_connections
                    .iter()
                    .filter(|&&(src, _dst, _)| src == $src)
                    .map(|v| (v.1, v.2))
            }};
            (PROP => $dst: expr) => {{
                prop_connections
                    .iter()
                    .filter(|&&(_src, dst, _)| dst == $dst)
                    .map(|v| (v.0, v.2))
            }};
        }

        let mut id_to_kv = HashMap::new();

        let objects = self
            .find_root("Objects")
            .into_iter()
            .flat_map(|o| &self.children[&o]);
        for &o in objects {
            let kv = &self.kvs[o];
            let id = kv.id().unwrap();
            let prev = id_to_kv.insert(id, o);
            assert_eq!(prev, None);
        }

        root_fields!(
          self,
          "Documents", &[],
          // I think this is only ever 1 for 1 scene
          "Count", &[Data::I32(1)] => |_| {},
          "Document", &[Data::I64(_), Data::String(_), Data::String(_)] => |c: usize| {
            let kv = &self.kvs[c];
            fbx_scene.id = kv.values[0].as_int().unwrap() as usize;
            assert_matches!(kv.values[1].as_str().unwrap(), "Scene" |  "");
            assert_eq!(kv.values[2].as_str().unwrap(), "Scene", "{kv:?}");
            match_children!(
              self, c,
              "Properties70", &[] => |c| match_children!(self,c,
                "P", &[
                  Data::String(_), Data::String(_), Data::String(_),
                  Data::I32(_) | Data::String(_)
                ] |
                &[
                  Data::String(_), Data::String(_), Data::String(_),
                  Data::String(_), Data::String(_)
                ] => |c: usize| {
                  let vals = &self.kvs[c].values;
                  match vals[0].as_str().unwrap() {
                    "SourceObject" => {},
                    "ActiveAnimStackName" => {},
                    x => todo_if_strict!("Unknown document property {x}"),
                  }
                }
              ),
              "RootNode", &[Data::I64(_)] => |c: usize| {
                match_children!(self, c);
                let id = *self.kvs[c].values[0].as_i64().unwrap();
                assert_eq!(None, id_to_kv.insert(id, c));
              },
            );
          },
        );

        let objects = self
            .find_root("Objects")
            .into_iter()
            .flat_map(|o| &self.children[&o]);
        for &o in objects {
            let kv = &self.kvs[o];
            let [id, name_objtype, classtag] = &kv.values[..] else {
                todo!("{:?}", kv.values);
            };
            let id = id.as_int().unwrap();
            let n_o = name_objtype.as_str().unwrap().split_once("\\x00\\x01");
            let Some((name, obj_type)) = n_o else {
                todo!("{name_objtype:?}");
            };

            let classtag = classtag.as_str().unwrap();

            match (kv.key.as_str(), obj_type) {
                ("NodeAttribute", "NodeAttribute") => match classtag {
                    "Light" => continue,
                    "Camera" => continue,
                    // I believe these are attributes for nodes, so they're not actual nodes.
                    // For now, no idea what to do with them.
                    "Null" => {
                        self.parse_null(id, id_to_kv[&id]);
                        assert_eq!(conns!(id =>).count(), 0);
                        for src in conns!(=> id) {
                            let node_idx = fbx_scene.node_by_id_or_new(src as usize);
                            fbx_scene.nodes[node_idx].is_null_node = true;
                        }
                    }
                    "LimbNode" => {
                        self.parse_limb_node(id, id_to_kv[&id]);
                        assert_eq!(conns!(id =>).count(), 0);
                        for src in conns!(=> id) {
                            let node_idx = fbx_scene.node_by_id_or_new(src as usize);
                            assert_eq!(fbx_scene.nodes[node_idx].limb_node_id, None);
                            fbx_scene.nodes[node_idx].limb_node_id = Some(id as usize);
                        }
                    }
                    _ => todo_if_strict!("NodeAttribute::{classtag} not handled"),
                },
                ("Geometry", "Geometry") => match classtag {
                    "Mesh" => {
                        let mi = fbx_scene.mesh_by_id_or_new(id as usize);
                        let fbx_mesh = &mut fbx_scene.meshes[mi];
                        self.parse_mesh(fbx_mesh, id, id_to_kv[&id]);
                        fbx_mesh.name = String::from(name);
                    }
                    // This is a blendshape
                    "Shape" => {
                        let bsi = fbx_scene.blendshape_by_id_or_new(id as usize);
                        let blendshape = &mut fbx_scene.blendshapes[bsi];
                        self.parse_blendshape(blendshape, id, id_to_kv[&id]);
                        blendshape.name = String::from(name);
                    }
                    _ => todo_if_strict!("Geometry::{classtag} not handled"),
                },
                // Not entirely sure when this is what, need to check more thoroughly
                ("Node" | "Model", "Model") => {
                    if matches!(classtag, "Light" | "Camera") {
                        continue;
                    }
                    assert_eq!(kv.key, "Model");

                    let node_idx = fbx_scene.node_by_id_or_new(id as usize);
                    let node = &mut fbx_scene.nodes[node_idx];
                    self.parse_node(node, id, id_to_kv[&id]);
                    node.name = String::from(name);

                    let mut num_parents = 0;
                    for parent_id in conns!(=> id) {
                        let Some(&kvi) = id_to_kv.get(&parent_id) else {
                            continue;
                        };
                        let parent = &self.kvs[kvi];
                        match parent.key.as_str() {
                            "RootNode" => fbx_scene.root_nodes.push(node_idx),
                            "Model" | "Node" => {
                                let parent_idx = fbx_scene.node_by_id_or_new(parent_id as usize);
                                fbx_scene.nodes[parent_idx].children.push(node_idx);
                                let node = &mut fbx_scene.nodes[node_idx];
                                assert_eq!(node.parent, None);
                                node.parent = Some(parent_idx);
                            }
                            "CollectionExclusive" => continue,
                            "Deformer" => continue,
                            x => todo!(
                                "Unknown parent for model {x:?} {:?}",
                                self.kvs[id_to_kv[&id]]
                            ),
                        }
                        num_parents += 1;
                    }

                    assert_eq!(num_parents, 1);

                    match classtag {
                        "Mesh" => {
                            assert_eq!(kv.key, "Model");
                            for c in conns!(id =>) {
                                let c_kv = &self.kvs[id_to_kv[&c]];
                                match c_kv.key.as_str() {
                                    "Geometry" => {
                                        let p = fbx_scene.mesh_by_id_or_new(c as usize);
                                        fbx_scene.nodes[node_idx].mesh = Some(p);
                                    }
                                    // Don't handle materials yet
                                    "Material" => {
                                        let p = fbx_scene.mat_by_id_or_new(c as usize);
                                        fbx_scene.nodes[node_idx].materials.push(p);
                                    }
                                    "Model" => {
                                        let n = fbx_scene.node_by_id_or_new(c as usize);
                                        fbx_scene.nodes[n].children.push(n);
                                    }
                                    x => todo!("{x:?}: {kv:?}"),
                                }
                            }
                        }
                        // FIXME handle these?
                        "Null" => {
                            assert_eq!(kv.key, "Model");
                            for dst_id in conns!(id =>) {
                                let dst = &self.kvs[id_to_kv[&dst_id]];
                                assert_matches!(
                                    dst.key.as_str(),
                                    "Model" | "Node" | "NodeAttribute"
                                );
                            }
                        }
                        "LimbNode" => {
                            assert_eq!(kv.key, "Model");
                            for dst_id in conns!(id =>) {
                                let dst = &self.kvs[id_to_kv[&dst_id]];
                                assert_matches!(
                                    dst.key.as_str(),
                                    "Model" | "Node" | "NodeAttribute"
                                );
                            }
                        }
                        x => todo_if_strict!("Unknown Model::classtag {x:?}"),
                    }
                }

                ("Material", "Material") => {
                    assert_eq!(classtag, "");
                    let mati = fbx_scene.mat_by_id_or_new(id as usize);
                    let mat = &mut fbx_scene.materials[mati];
                    self.parse_material(mat, id, id_to_kv[&id]);
                    mat.name = String::from(name);
                }
                ("Deformer", "Deformer") => match classtag {
                    "Skin" => {
                        let skin_idx = fbx_scene.skin_by_id_or_new(id as usize);
                        let skin = &mut fbx_scene.skins[skin_idx];
                        self.parse_skin(skin, id, id_to_kv[&id]);
                        skin.name = String::from(name);
                        for src in conns!(=> id) {
                            assert_eq!("Geometry", self.kvs[id_to_kv[&src]].key);
                            let mesh_idx = fbx_scene.mesh_by_id_or_new(src as usize);
                            fbx_scene.skins[skin_idx].mesh = mesh_idx;
                            assert_eq!(fbx_scene.meshes[mesh_idx].skin, None);
                            fbx_scene.meshes[mesh_idx].skin = Some(skin_idx);
                        }
                        for dst in conns!(id =>) {
                            assert_eq!("Deformer", self.kvs[id_to_kv[&dst]].key);
                        }
                    }
                    "BlendShape" => {
                        let bsi = fbx_scene.blendshape_by_id_or_new(id as usize);
                        let blendshape = &mut fbx_scene.blendshapes[bsi];
                        self.parse_blendshape(blendshape, id, id_to_kv[&id]);
                        blendshape.name = String::from(name);
                    }
                    _ => todo_if_strict!("Unknown deformer {classtag}"),
                },
                ("Deformer", "SubDeformer") => match classtag {
                    "Cluster" => {
                        let cl_idx = fbx_scene.cluster_by_id_or_new(id as usize);
                        self.parse_cluster(&mut fbx_scene.clusters[cl_idx], id, id_to_kv[&id]);
                        fbx_scene.clusters[cl_idx].name = String::from(name);
                        for src in conns!(=> id) {
                            assert_eq!("Deformer", self.kvs[id_to_kv[&src]].key);
                            let skin_idx = fbx_scene.skin_by_id_or_new(src as usize);
                            fbx_scene.skins[skin_idx].clusters.push(cl_idx);

                            assert_eq!(fbx_scene.clusters[cl_idx].skin, 0);
                            fbx_scene.clusters[cl_idx].skin = skin_idx;
                        }
                        for dst in conns!(id =>) {
                            assert_matches!(
                                self.kvs[id_to_kv[&dst]].key.as_str(),
                                "Node" | "Model" /* | "Geometry" // TODO */
                            );
                            let node_idx = fbx_scene.node_by_id_or_new(dst as usize);
                            assert_eq!(fbx_scene.nodes[node_idx].cluster, None);
                            fbx_scene.nodes[node_idx].cluster = Some(cl_idx);
                            assert_eq!(fbx_scene.clusters[cl_idx].node, 0);
                            fbx_scene.clusters[cl_idx].node = node_idx;
                        }
                    }
                    "BlendShapeChannel" => {
                        let bs_ch_idx = fbx_scene.blendshape_channel_by_id_or_new(id as usize);
                        let bs_ch = &mut fbx_scene.blendshape_channels[bs_ch_idx];
                        self.parse_blendshape_channel(bs_ch, id, id_to_kv[&id]);
                        for src in conns!(=> id) {
                            assert_eq!("Deformer", self.kvs[id_to_kv[&src]].key);
                            assert_eq!(
                                Some("BlendShape"),
                                self.kvs[id_to_kv[&src]].values[2].as_str()
                            );
                            let bs_idx = fbx_scene.blendshape_by_id_or_new(src as usize);
                            fbx_scene.blendshapes[bs_idx].channels.push(bs_ch_idx);
                        }
                        for dst in conns!(id =>) {
                            assert_eq!("Geometry", self.kvs[id_to_kv[&dst]].key);
                            let mesh_idx = fbx_scene.mesh_by_id_or_new(dst as usize);
                            assert_eq!(fbx_scene.blendshape_channels[bs_ch_idx].mesh, 0);
                            fbx_scene.blendshape_channels[bs_ch_idx].mesh = mesh_idx;
                        }
                    }
                    _ => todo_if_strict!("Unknown subdeformer {classtag}"),
                },
                ("Implementation", "Implementation") => {
                    assert_eq!(classtag, "");
                }
                ("AnimationStack", "AnimStack") => {
                    assert_eq!(classtag, "");
                    let as_idx = fbx_scene.anim_stack_by_id_or_new(id as usize);
                    let anim_stack = &mut fbx_scene.anim_stacks[as_idx];
                    anim_stack.name = String::from(name);
                    self.parse_anim_stack(anim_stack, id, id_to_kv[&id]);
                    for dst in conns!(id =>) {
                        let dst_kv = &self.kvs[id_to_kv[&dst]];
                        assert_eq!(dst_kv.key, "AnimationLayer");
                        let al_idx = fbx_scene.anim_layer_by_id_or_new(dst as usize);
                        fbx_scene.anim_layers[al_idx].anim_stack = as_idx;
                    }
                    assert_eq!(conns!(=> id).count(), 0);
                }
                ("AnimationLayer", "AnimLayer") => {
                    assert_eq!(classtag, "");
                    let anim_layer_idx = fbx_scene.anim_layer_by_id_or_new(id as usize);
                    let anim_layer = &mut fbx_scene.anim_layers[anim_layer_idx];
                    self.parse_anim_layer(anim_layer, id, id_to_kv[&id]);
                    anim_layer.name = String::from(name);
                }
                ("AnimationCurveNode", "AnimCurveNode") => {
                    assert_eq!(classtag, "");
                    let acn_idx = fbx_scene.anim_curve_node_by_id_or_new(id as usize);
                    let anim_curve_node = &mut fbx_scene.anim_curve_nodes[acn_idx];
                    anim_curve_node.name = String::from(name);
                    self.parse_anim_curve_node(anim_curve_node, id, id_to_kv[&id]);
                    assert_eq!(conns!(id =>).count(), 0);
                    for src in conns!(=> id) {
                        let kv = &self.kvs[id_to_kv[&src]];
                        assert_eq!(kv.key, "AnimationLayer");
                        let al_idx = fbx_scene.anim_layer_by_id_or_new(src as usize);
                        fbx_scene.anim_curve_nodes[acn_idx].layer = al_idx;
                    }
                }
                ("AnimationCurve", "AnimCurve") => {
                    assert_eq!(classtag, "");
                    let ac_idx = fbx_scene.anim_curve_by_id_or_new(id as usize);
                    let anim = &mut fbx_scene.anim_curves[ac_idx];
                    self.parse_anim_curve(anim, id, id_to_kv[&id]);
                    assert_eq!(conns!(id =>).count(), 0);
                    assert_eq!(conns!(=> id).count(), 0);
                    // has property connections
                    for (dst, key) in conns!(PROP => id) {
                        assert_matches!(key, "d|X" | "d|Y" | "d|Z" | "d|DeformPercent");
                        assert_eq!("AnimationCurveNode", self.kvs[id_to_kv[&dst]].key);
                        assert_eq!(0, fbx_scene.anim_curves[ac_idx].anim_curve_node);
                        let acn_idx = fbx_scene.anim_curve_node_by_id_or_new(id as usize);
                        // TODO is this inverted?
                        assert_eq!(fbx_scene.anim_curves[ac_idx].anim_curve_node, 0);
                        fbx_scene.anim_curves[ac_idx].anim_curve_node = acn_idx;
                    }
                }
                // Don't handle these yet
                ("Texture", "Texture") => {
                    let tex_id = fbx_scene.texture_by_id_or_new(id as usize);
                    let tex = &mut fbx_scene.textures[tex_id];
                    self.parse_texture(tex, id, id_to_kv[&id]);
                    tex.name = String::from(name);
                }
                ("DisplayLayer", "DisplayLayer") => continue,
                ("Video", "Video") => continue,
                ("Light", "Light") => continue,
                ("BindingTable", "BindingTable") => continue,

                ("Pose", "Pose") => {
                    assert_eq!(classtag, "BindPose");
                    self.parse_pose(&mut fbx_scene, id, id_to_kv[&id]);
                    let pose_idx = fbx_scene.pose_by_id_or_new(id as usize);
                    fbx_scene.poses[pose_idx].name = String::from(name);
                    assert_eq!(conns!(id =>).count(), 0);
                    assert_eq!(conns!(=> id).count(), 0);

                    assert_eq!(conns!(PROP id =>).count(), 0);
                    assert_eq!(conns!(PROP => id).count(), 0);
                }
                ("LayeredTexture", "LayeredTexture") => continue,

                (key, ty) => todo_if_strict!("Unknown key object type {key} {ty}"),
            }
        }

        /*
        for &(parent, child, key) in &prop_connections {
            println!(
                "{:?} {:?} (key = {key})",
                fbx_scene.id_kind(parent as usize),
                fbx_scene.id_kind(child as usize),
            );
        }
        */

        root_fields!(
          self,
          "FBXHeaderExtension", &[],
          "FBXHeaderVersion", &[Data::I32(_)] => |_| {},
          "FBXVersion", &[Data::I32(_)] => |_| {},
          "EncryptionType", &[Data::I32(0)] => |_| {},
          "CreationTimeStamp", &[] => |c| match_children!(
            self, c,
            "Version", &[Data::I32(_)] => |_| {},
            "Year", &[Data::I32(_)] => |_| {},
            "Month", &[Data::I32(_)] => |_| {},
            "Day", &[Data::I32(_)] => |_| {},
            "Hour", &[Data::I32(_)] => |_| {},
            "Minute", &[Data::I32(_)] => |_| {},
            "Second", &[Data::I32(_)] => |_| {},
            "Millisecond", &[Data::I32(_)] => |_| {},
          ),
          "Creator", &[Data::String(_)] => |_| {},
          "SceneInfo", &[Data::String(_), Data::String(_)] => |c: usize| {
            match_children!(
              self, c,
              "Type", &[Data::String(_)] => |c: usize| {
                assert_eq!("UserData", self.kvs[c].values[0].as_str().unwrap());
              },
              "Version", &[Data::I32(_)] => |_| {},
              "MetaData", &[] => |c| match_children!(
                self, c,
                "Version", &[Data::I32(_)] => |_| {},
                "Title", &[Data::String(_)] => |_| {},
                "Subject", &[Data::String(_)] => |_| {},
                "Author", &[Data::String(_)] => |_| {},
                "Keywords", &[Data::String(_)] => |_| {},
                "Revision", &[Data::String(_)] => |_| {},
                "Comment", &[Data::String(_)] => |_| {},
              ),
              "Properties70", &[] => |_| { /* match_children!(self, v) */ },
            );
          },

          "OtherFlags", &[] => |_| {},
        );

        if let Some(file_id) = self.find_root("FileId") {
            let kv = &self.kvs[file_id];
            assert_matches!(kv.values.as_slice(), &[Data::Binary(_)]);
            let Data::Binary(ref b) = kv.values[0] else {
                unreachable!();
            };
            assert_eq!(b.len(), 16);
            fbx_scene.file_id[..].clone_from_slice(b);
        } else {
            eprintln!("Missing File ID in FBX");
        }

        root_fields!(self, "CreationTime", &[Data::String(_)]);
        root_fields!(self, "Creator", &[Data::String(_)]);

        let settings = &mut fbx_scene.global_settings;

        root_fields!(
          self,
          "GlobalSettings", &[],
          "Version", &[Data::I32(_)] => |_| {},
          "Properties70", &[] => |c| {
            match_children!(
              self, c,
              "P",
              &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::I32(_) | Data::I64(_) | Data::F64(_) | Data::String(_),
              ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
                Data::F64(_), Data::F64(_), Data::F64(_)
              ] | &[
                Data::String(_), Data::String(_), Data::String(_), Data::String(_),
              ] => |c: usize| {
                let vals = &self.kvs[c].values;
                macro_rules! assign {
                  ($v: expr, $fn: ident) => {{
                    $v = *vals[4].$fn().unwrap_or(&$v);
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
                  "AmbientColor" => {},
                  "DefaultCamera" => {},
                  "TimeMode" => {},
                  "TimeSpanStart" => {},
                  "TimeSpanStop" => {},
                  "CustomFrameRate" => {},
                  "TimeProtocol" => {},
                  "SnapOnFrameMode" => {},
                  "TimeMarker" => {},
                  "CurrentTimeMarker" => {},
                  x => todo!("Unhandled Properties70 P {x:?}")
                };

              },
            );
          },
        );

        // TODO what is this for?
        root_fields!(self, "References", &[]);

        root_fields!(
          self,
          "Definitions", &[],
          "Version", &[Data::I32(_)] => |_| {},
          "Count", &[Data::I32(_)] => |_| {},
          "ObjectType", &[Data::String(_)] => |c| match_children!(self, c,
            "Count", &[Data::I32(_)] => |_| {},
            "PropertyTemplate", &[Data::String(_)] => |c: usize| assert_matches!(
              self.kvs[c].values[0].as_str().unwrap(),
              "FbxVideo" | "FbxAnimCurveNode" | "FbxFileTexture" | "FbxBindingTable" |
              "FbxImplementation" | "FbxNull" | "FbxSurfaceLambert" | "FbxAnimLayer" |
              "FbxAnimStack" | "FbxCamera" | "FbxMesh" | "FbxNode" | "FbxSurfacePhong" |
              "FbxDisplayLayer" | "FbxLayeredTexture" | "FbxLight"

            ),
          ),
        );

        // objects (handled earlier)
        // conections (handled earlier)

        root_fields!(
          self,
          "Takes", &[],
          "Current", &[Data::String(_)] => |_| {},
          "Take", &[Data::String(_)] => |_| {}
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
    let block_len = end_offset.saturating_sub(prev_read as u64);

    let prop_count = read_word!();
    let prop_len = read_word!();

    let scope_name = read_string!(false, false);

    // while technically this could be right under where end_offset is read,
    // some parsers expect all the above properties to be included.
    // By having it here, it ensures that all the above are properly set.
    if end_offset == 0 {
        return Ok((false, read));
    }
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
                src.seek(SeekFrom::Current(len as i64))?;
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
