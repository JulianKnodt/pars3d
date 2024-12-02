use super::parser::{Data, Token, KV};
use super::{FBXMesh, FBXNode, FBXScene};
use std::io::{self, Seek, SeekFrom, Write};

use std::collections::{HashMap, HashSet};

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

macro_rules! push_kv {
    ($kvs: expr, $kv: expr) => {{
        let idx = $kvs.len();
        $kvs.push($kv);
        idx
    }};
}

impl FBXScene {
    pub(crate) fn to_kvs(&self) -> Vec<KV> {
        let mut kvs = vec![];
        let conn_idx = push_kv!(kvs, KV::new("Connections", &[], None));
        // for each node add a connection from it to its parent
        for ni in 0..self.nodes.len() {
            let parent = self.parent_node(ni);
            let id = match parent {
                None => 0,
                Some(p) => self.nodes[p].id,
            };
            let own_id = self.nodes[ni].id;
            let vals = &[
                Data::str("OO"),
                Data::I64(own_id as i64),
                Data::I64(id as i64),
            ];
            push_kv!(kvs, KV::new("C", vals, Some(conn_idx)));
        }

        let obj_kv = push_kv!(kvs, KV::new("Objects", &[], None));

        for mesh in &self.meshes {
            mesh.to_kvs(obj_kv, &mut kvs);
        }

        for node in &self.nodes {
            node.to_kvs(obj_kv, &mut kvs);
        }

        kvs
    }
}

impl FBXMesh {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let vals = [
            Data::I64(self.id as i64),
            Data::str("\x00\x01Geometry"),
            Data::str("Mesh"),
        ];
        let mesh_kv = push_kv!(kvs, KV::new("Geometry", &vals, Some(parent)));

        push_kv!(
            kvs,
            KV::new("GeometryVersion", &[Data::I32(101)], Some(mesh_kv))
        );

        let vert_vals = self
            .v
            .iter()
            .flat_map(|v| v.iter().map(|&v| v as f64))
            .collect::<Vec<f64>>();

        push_kv!(
            kvs,
            KV::new("Vertices", &[Data::F64Arr(vert_vals)], Some(mesh_kv))
        );

        let faces = self
            .f
            .iter()
            .flat_map(|f| {
                let (&last, rest) = f.as_slice().split_last().unwrap();
                let last = last as i32;
                rest.iter()
                    .map(|&v| v as i32)
                    .chain(std::iter::once(-last - 1))
            })
            .collect::<Vec<i32>>();

        push_kv!(
            kvs,
            KV::new("PolygonVertexIndex", &[Data::I32Arr(faces)], Some(mesh_kv))
        );

        // TODO export UV and normals
    }
}

impl FBXNode {
    fn to_kvs(&self, parent: usize, kvs: &mut Vec<KV>) {
        let vals = [
            Data::I64(self.id as i64),
            Data::String(format!("{}\x00\x01Model", self.name)),
            Data::str("Mesh"),
        ];

        let _node_kv = push_kv!(kvs, KV::new("Model", &vals, Some(parent)));
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
                x => todo!("{x:?}"),
            };
            let c = write_word!($dst, u8, c);
            assert_eq!(c, 1);
            c + match $d {
                Data::I32(i) => write_word!($dst, i32, *i),
                Data::I64(i) => write_word!($dst, i64, *i),
                Data::I64Arr(arr) => write_arr!($dst, i64, arr),
                Data::I32Arr(arr) => write_arr!($dst, i32, arr),
                Data::F64Arr(arr) => write_arr!($dst, f64, arr),
                Data::String(s) => write_string!($dst, s, true, true),
                _ => todo!(),
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
    assert!(!matches!(tokens[i], Token::Data(_)));
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
    let _ = w.write(&(0u64).to_le_bytes())?;

    Ok(())
}
