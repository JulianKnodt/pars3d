use std::path::{Path, PathBuf};

use std::io;

/// Computes the relative path from src to dst.
pub fn rel_path_btwn(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> io::Result<PathBuf> {
    let src: &Path = src.as_ref();
    let dst: &Path = dst.as_ref();
    let abs_src = src.canonicalize()?;
    let abs_dst = dst.canonicalize()?;

    let mut curr = if abs_src.is_file() {
        abs_src.parent()
    } else {
        Some(abs_src.as_path())
    };
    let mut num_parents = 0;
    while let Some(c) = curr
        && !abs_dst.starts_with(c)
    {
        num_parents += 1;
        curr = c.parent();
    }

    let prefix = match curr {
        None => panic!(
            "No relative path between absolute paths {}->{}",
            abs_src.display(),
            abs_dst.display()
        ),
        Some(c) => abs_dst.strip_prefix(c).unwrap(),
    };
    let mut backs = PathBuf::new();
    backs.extend((0..num_parents).map(|_| ".."));
    backs.push(prefix);
    Ok(backs)
}

/// File formats supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    // Primary support
    /// GLTF Binary Format (.glb)
    GLB,
    /// FBX Binary Format (.fbx)
    FBX,
    /// OBJ Binary Format (.obj), mtl not included
    OBJ,

    // Secondary Support
    /// PLY file format (.ply)
    PLY,
    /// STL file format (.stl)
    STL,
    /// OFF file format (.off)
    OFF,

    /// Other unsupported file formats
    Unknown,
}

/// Given something that looks like a path parse it into a FileFormat.
pub fn extension_to_format(s: impl AsRef<Path>) -> FileFormat {
    let s = s.as_ref();
    let Some(e) = s.extension() else {
        return FileFormat::Unknown;
    };
    let Some(e) = e.to_str() else {
        return FileFormat::Unknown;
    };

    let matches = [
        ("glb", FileFormat::GLB),
        ("fbx", FileFormat::FBX),
        ("obj", FileFormat::OBJ),
        ("ply", FileFormat::PLY),
        ("stl", FileFormat::STL),
        ("off", FileFormat::OFF),
    ];
    for (ext, fmt) in matches {
        if ext.eq_ignore_ascii_case(e) {
            return fmt;
        }
    }
    FileFormat::Unknown
}

#[ignore]
#[test]
fn test_basic() {
    let v = rel_path_btwn("test/a/b", "b").unwrap();
    let exp: &Path = ("../../../b").as_ref();
    assert_eq!(v, exp);

    let v = rel_path_btwn("b", "test/a/b").unwrap();
    let exp: &Path = ("../test/a/b").as_ref();
    assert_eq!(v, exp);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JSONPrimitive<'a> {
    String(&'a str),
    Number(f64),
    Bool(bool),
    Null,
}

impl std::fmt::Display for JSONPrimitive<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        use JSONPrimitive::*;
        match self {
            String(s) => write!(f, "\"{s}\""),
            Number(n) => write!(f, "{n}"),
            Bool(b) => write!(f, "{b}"),
            Null => write!(f, "null"),
        }
    }
}

impl<'a> Into<JSONPrimitive<'a>> for &'a str {
    fn into(self) -> JSONPrimitive<'a> {
        JSONPrimitive::String(self)
    }
}
impl<'a> Into<JSONPrimitive<'a>> for bool {
    fn into(self) -> JSONPrimitive<'a> {
        JSONPrimitive::Bool(self)
    }
}
impl<'a> Into<JSONPrimitive<'a>> for f32 {
    fn into(self) -> JSONPrimitive<'a> {
        JSONPrimitive::Number(self as f64)
    }
}
impl<'a> Into<JSONPrimitive<'a>> for f64 {
    fn into(self) -> JSONPrimitive<'a> {
        JSONPrimitive::Number(self)
    }
}
impl<'a> Into<JSONPrimitive<'a>> for usize {
    fn into(self) -> JSONPrimitive<'a> {
        JSONPrimitive::Number(self as f64)
    }
}

/// Append a key value pair to a given JSON.
pub fn append_json<'a, 'b>(
    s: &mut String,
    indent: usize,
    k: impl Into<JSONPrimitive<'a>>,
    v: impl Into<JSONPrimitive<'b>>,
) {
    let mut saw_bracket = false;
    let mut is_start = false;
    while let Some(l) = s.pop() {
        if l == '}' {
            saw_bracket = true;
            continue;
        }

        if !l.is_whitespace() && saw_bracket {
            is_start = l == '{';
            s.push(l);
            break;
        }
    }
    if !is_start {
        s.push(',');
    }
    s.push('\n');
    for _ in 0..indent {
        s.push(' ');
    }
    use std::fmt::Write;
    let k = k.into();
    let v = v.into();
    write!(s, "{k}: {v}\n}}").unwrap();
}

#[test]
fn test_append_json() {
    let mut v = String::from("{}");
    append_json(&mut v, 2, "test", 2);
    assert_eq!(v, "{\n  \"test\": 2\n}");
    append_json(&mut v, 2, "other", "derp");
    assert_eq!(v, "{\n  \"test\": 2,\n  \"other\": \"derp\"\n}");
}
