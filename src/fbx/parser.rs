use super::FBXScene;
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

#[derive(Debug, Clone, Default)]
struct KVs {
    kvs: Vec<KV>,
    roots: Vec<usize>,
}

impl KVs {
    // parses the token stream until a scope end.
    // returns the index of the newly produced datablock.
    fn parse_scope(&mut self, tokens: &mut impl Iterator<Item = Token>, parent: Option<usize>) {
        let mut i = self.kvs.len();
        self.kvs.push(Default::default());
        self.kvs[i].parent = parent;
        if parent.is_none() {
            self.roots.push(i);
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
                    }
                }
                Token::ScopeStart => self.parse_scope(tokens, Some(i)),
                Token::Data(d) => self.kvs[i].values.push(d),
                Token::ScopeEnd => return,
            }
        }
    }

    // TODO reconvert KVs to tokens

    /// Constructs a graphviz representation of this FBX file, for viewing externally
    fn to_graphviz(&self, mut dst: impl Write) -> io::Result<()> {
        let mut rev_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, kv) in self.kvs.iter().enumerate() {
            let Some(p) = kv.parent else { continue };
            rev_map.entry(p).or_default().push(i);
        }

        writeln!(dst, "graph FBX {{")?;
        for (i, kv) in self.kvs.iter().enumerate() {
            writeln!(dst, "\t{i} [label=\"{}\"]", kv.key)?;
        }
        for (k, vs) in rev_map.into_iter() {
            for v in vs.into_iter() {
                writeln!(dst, "\t{k} -- {v}")?;
            }
        }
        writeln!(dst, "}}")?;
        Ok(())
    }

    fn to_scene(&self) -> FBXScene {
        let mut rev_map: HashMap<usize, Vec<usize>> = HashMap::new();

        for (i, kv) in self.kvs.iter().enumerate() {
            let Some(p) = kv.parent else { continue };
            rev_map.entry(p).or_default().push(i);
        }
        let mut fbx_scene = FBXScene::default();

        for &i in rev_map.keys() {
          println!("{:?}", self.kvs[i]);
        }
        //let mut connections = vec![];
        let conn_kv = self.roots
            .iter()
            .find(|&&v| self.kvs[v].key == "Connections")
            .expect("No connections?");

        for &child in &rev_map[conn_kv] {
          println!("{:?}", self.kvs[child]);
        }

        //let mut objects = vec![];
        todo!();
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

    println!("{prop_count} {prop_len} {scope_name} {block_len}");

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

    let scene = kvs.to_scene();

    todo!();
}
