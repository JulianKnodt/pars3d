#![allow(unused)]

use super::{FBXMesh, FBXScene};
use crate::{FaceKind, F};

use std::ascii::Char;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::mem::size_of;

/// Magic binary length.
const MAGIC_LEN: usize = 23;

/// Magic binary.
pub(crate) const MAGIC: &[u8; MAGIC_LEN] = b"Kaydara FBX Binary  \x00\x1a\x00";

#[derive(Debug, Clone)]
pub enum Data {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),

    U32(u32),

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

    fn as_int(&self) -> Option<i64> {
        match self {
            &Data::I64(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum Token {
    Key(String),
    Data(Data),
    ScopeStart,
    ScopeEnd,
}

#[derive(Debug, Clone, Default)]
struct KV {
    key: String,
    values: Vec<Data>,
    parent: Option<usize>,
}

impl KV {
    pub fn id(&self) -> Option<i64> {
        self.values.get(0).and_then(Data::as_int)
    }
}

#[derive(Debug, Clone, Default)]
struct KVs {
    kvs: Vec<KV>,
    roots: Vec<usize>,
    children: HashMap<usize, Vec<usize>>,
}

impl KVs {
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

    // TODO reconvert KVs to tokens

    /*
    /// Constructs a graphviz representation of this FBX file, for viewing externally
    #[allow(unused)]
    pub fn to_graphviz(&self, mut dst: impl Write) -> io::Result<()> {
        writeln!(dst, "graph FBX {{")?;
        for (i, kv) in self.kvs.iter().enumerate() {
            writeln!(dst, "\t{i} [label=\"{}\"]", kv.key)?;
        }
        for &(k, vs) in self.children.iter() {
            for &v in vs.iter() {
                writeln!(dst, "\t{k} -- {v}")?;
            }
        }
        writeln!(dst, "}}")?;
        Ok(())
    }
    */
    fn parse_mesh(&self, mesh_id: i64, kvi: usize) -> FBXMesh {
        let mut out = FBXMesh::default();
        for &c in &self.children[&kvi] {
            let child = &self.kvs[c];
            match child.key.as_str() {
                "Vertices" => {
                    // TODO or this can be f32?
                    let v_arr: &[f64] = child.values[0].as_f64_arr().unwrap();
                    let v = v_arr
                        .iter()
                        .array_chunks::<3>()
                        .map(|[a, b, c]| [*a as F, *b as F, *c as F]);
                    out.v.extend(v);
                }
                "Properties70" => {
                    assert!(child.values.is_empty());
                }
                "GeometryVersion" => {}
                "PolygonVertexIndex" => {
                    let mut curr_face = FaceKind::empty();
                    let idxs = child.values[0].as_i32_arr().unwrap();
                    for &vi in idxs {
                        if vi >= 0 {
                            curr_face.insert(vi as usize);
                        } else {
                            curr_face.insert(-(vi + 1) as usize);
                            let f = std::mem::replace(&mut curr_face, FaceKind::empty());
                            out.f.push(f);
                        }
                    }
                }
                "Edges" => { /* No idea what to do here */ }
                "LayerElementNormal" => {
                    for &cc in &self.children[&c] {
                        let gc = &self.kvs[cc];
                        match gc.key.as_str() {
                            "Version" => {}
                            "Name" => {}
                            "MappingInformationType" => {}
                            "ReferenceInformationType" => {}
                            "Normals" => todo!(),
                        }
                    }
                }
                "LayerElementUV" => {
                    for &cc in &self.children[&c] {
                        let gc = &self.kvs[cc];
                        match gc.key.as_str() {
                            "Version" => {}
                            "Name" => {}
                            "MappingInformationType" => {}
                            "ReferenceInformationType" => {}
                            "UV" => todo!(),
                            "UVIndex" => todo!(),
                        }
                    }
                }
                "LayerElementMaterial" => {
                    for &cc in &self.children[&c] {
                        let gc = &self.kvs[cc];
                        match gc.key.as_str() {
                            "Version" => {}
                            "Name" => {}
                            "MappingInformationType" => {}
                            "ReferenceInformationType" => {}
                            "UV" => todo!(),
                            "UVIndex" => todo!(),
                        }
                    }
                }
                x => todo!("{x:?} {:?}", child.values),
            }
        }
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
        let conn_idx = self
            .roots
            .iter()
            .find(|&&v| self.kvs[v].key == "Connections");
        if let Some(conn_idx) = conn_idx {
            for &child in &self.children[conn_idx] {
                let kv = &self.kvs[child];
                assert_eq!(kv.key, "C");
                assert_eq!(kv.values.len(), 3);
                let [marker, src, dst] = &kv.values[..] else {
                    todo!("{:?}", kv.values);
                };
                assert_eq!(marker.as_str().unwrap(), "OO", "Temporary check {marker:?}");
                connections.push((src.as_int().unwrap(), dst.as_int().unwrap()));
            }
        }

        let mut objects = self.roots.iter().find(|&&v| self.kvs[v].key == "Objects");
        let objects = objects.into_iter().flat_map(|o| &self.children[o]);
        for &o in objects {
            let kv = &self.kvs[o];
            let [id, name_objtype, classtag] = &kv.values[..] else {
                todo!("{:?}", kv.values);
            };
            let id = id.as_int().unwrap();
            assert_eq!(kv.values.len(), 3);
            let n_o = name_objtype.as_str().unwrap().split_once("\\x00\\x01");
            let Some((name, obj_type)) = n_o else {
                todo!("{name_objtype:?}");
            };

            let Some(classtag) = classtag.as_str() else {
                todo!("{classtag:?}");
            };

            let out_object = match obj_type {
                "NodeAttribute" => match classtag {
                    "Light" => continue,
                    "Camera" => continue,
                    _ => todo!("NodeAttribute::{classtag} not handled"),
                },
                "Geometry" => match classtag {
                    "Mesh" => {
                        let fbx_mesh = self.parse_mesh(id, id_to_kv[&id]);
                        fbx_scene.meshes.push(fbx_mesh);
                    }
                    _ => todo!("Geometry::{classtag} not handled"),
                },
                // Do not handle lights or cameras for now
                /*
                "Model" => match classtag {
                  "Geometry"
                },
                */
                // Don't handle materials yet
                "Material" => continue,
                _ => todo!("{obj_type:?}"),
            };
        }

        todo!("Where we at");

        fbx_scene
    }
}

fn parse_tokens(mut tokens: impl Iterator<Item = Token>) -> KVs {
    let mut kvs = KVs::default();
    kvs.parse_scope(&mut tokens, None);
    kvs
}

pub fn tokenize_binary(mut src: impl BufRead) -> io::Result<Vec<Token>> {
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
    src: &mut impl BufRead,
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
            src.read_exact(&mut v)?;
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
            assert_eq!(len * stride, comp_len);
            let mut out = vec![];
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
            let mut out = vec![];
            out.reserve(len);
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

    if end_offset == 0 {
        return Ok((false, read));
    }
    let block_len = end_offset - (prev_read as u64);

    let prop_count = read_word!();
    let prop_len = read_word!();
    let scope_name = read_string!(false, false);

    output_tokens.push(Token::Key(scope_name));

    let curr_read = read;
    for _pi in 0..prop_count {
        let data = match read_word!(u8).as_ascii() {
            None => todo!(),
            // TODO are these signed or unsigned?
            Some(Char::CapitalY) => Data::I16(read_word!(i16)),
            Some(Char::CapitalI) => Data::I32(read_word!(i32)),
            Some(Char::CapitalL) => Data::I64(read_word!(i64)),

            Some(Char::CapitalF) => Data::F32(read_word!(f32)),
            Some(Char::CapitalD) => Data::F64(read_word!(f64)),
            Some(Char::CapitalR) => {
                let len = read_word!(u32);
                Data::Binary(read_buf!(len as usize))
            }

            Some(Char::SmallF) => Data::F32Arr(read_array!(f32)),
            Some(Char::SmallD) => Data::F64Arr(read_array!(f64)),
            Some(Char::SmallI) => Data::I32Arr(read_array!(i32)),
            Some(Char::SmallL) => Data::I64Arr(read_array!(i64)),
            Some(Char::SmallC) => Data::BoolArr(read_array!(bool)),

            Some(Char::CapitalS) => Data::String(read_string!(true, true)),
            Some(Char::CapitalC) => Data::Bool(read_word!(bool)),
            Some(Char::SmallB) => todo!("Unknown how to handle small b"),

            Some(c) => todo!("unhandled {c:?} (u8 = {})", c.to_u8()),
        };
        output_tokens.push(Token::Data(data));
    }
    assert_eq!((read - curr_read) as u64, prop_len);

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
        output_tokens.push(Token::ScopeEnd);

        let sentinel = read_word!(sentinel_block_len as usize);
        assert!(sentinel.iter().all(|&v| v == b'\0'));
    }

    assert_eq!(read as u64, block_len);

    Ok((true, read))
}

#[test]
fn test_parse_fbx() {
    use std::fs::File;
    use std::io::BufReader;
    let f = File::open("cube.fbx").unwrap();
    let tokens = tokenize_binary(BufReader::new(f)).expect("Failed to tokenize FBX");
    let kvs = parse_tokens(tokens.into_iter());

    /*
    let vis = File::create("fbx.dot").expect("Failed to create dot file for viewing FBX");
    kvs.to_graphviz(io::BufWriter::new(vis))
        .expect("Failed to write graphviz");
    */

    let _scene = kvs.to_scene();

    todo!();
}
