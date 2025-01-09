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

#[test]
fn test_basic() {
    let v = rel_path_btwn("test/a/b", "b").unwrap();
    let exp: &Path = ("../../../b").as_ref();
    assert_eq!(v, exp);

    let v = rel_path_btwn("b", "test/a/b").unwrap();
    let exp: &Path = ("../test/a/b").as_ref();
    assert_eq!(v, exp);
}
