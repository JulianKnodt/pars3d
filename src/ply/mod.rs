use super::{FaceKind, F};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Error, ErrorKind, Read, Write};
use std::path::Path;

pub mod to_mesh;

/// A PLY mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Ply {
    /// Vertex positions
    v: Vec<[F; 3]>,
    /// Vertex colors
    vc: Vec<[u8; 3]>,

    f: Vec<FaceKind>,
}

#[derive(PartialEq)]
enum ReadExpect {
    Header,
    Format,
    VertexCount,
    VertexProperty,
    PropertyList,
    EndHeader,
    Vertices,
    Faces,
    Done,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Type {
    UChar,
    Float,
}

enum Field {
    X,
    Y,
    Z,
    Red,
    Green,
    Blue,
    Alpha,
}

impl Ply {
    /// Construct a new PLY mesh, with optional empty vertex colors.
    pub fn new(v: Vec<[F; 3]>, vc: Vec<[u8; 3]>, f: Vec<FaceKind>) -> Self {
        assert!(vc.is_empty() || v.len() == vc.len());
        Self { v, vc, f }
    }

    pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<Self> {
        Self::read(File::open(p)?)
    }

    pub fn read(r: impl Read) -> io::Result<Self> {
        Self::buf_read(BufReader::new(r))
    }
    pub fn buf_read(r: impl BufRead) -> io::Result<Self> {
        let mut state = ReadExpect::Header;
        let mut prop_set = vec![];
        let mut fields = vec![];
        let mut has_color = false;

        let mut v = vec![];
        let mut vc = vec![];
        let mut f = vec![];
        let mut num_v = 0;
        let mut num_f = 0;
        macro_rules! parse_err {
            ($s: expr) => {
                Err(Error::new(ErrorKind::Other, $s))
            };
        }

        for l in r.lines() {
            let l = l?;
            use ReadExpect::*;
            state = match state {
                Header => {
                    if l != "ply" {
                        return parse_err!("Expected ply as 1st line.");
                    } else {
                        Format
                    }
                }
                Format => {
                    if l != "format ascii 1.0" {
                        return parse_err!("Unsupported ply format (only ascii supported)");
                    } else {
                        VertexCount
                    }
                }
                VertexCount => {
                    let mut tokens = l.split_whitespace();
                    if tokens.next() != Some("element") {
                        return parse_err!("Missing 'element'");
                    }
                    if tokens.next() != Some("vertex") {
                        return parse_err!("Missing 'vertex'");
                    }
                    let Some(vc) = tokens.next() else {
                        return parse_err!("Missing vertex count");
                    };
                    let Ok(vc) = vc.parse::<usize>() else {
                        return parse_err!("Vertex count could not be parsed");
                    };
                    v.reserve(vc);
                    num_v = vc;

                    VertexProperty
                }
                VertexProperty if l.starts_with("element") => {
                    let mut tokens = l.split_whitespace();

                    if tokens.next() != Some("element") {
                        return parse_err!("Missing 'element'");
                    }
                    if tokens.next() != Some("face") {
                        return parse_err!("Missing 'face'");
                    }
                    let Some(fc) = tokens.next() else {
                        return parse_err!("Missing face count");
                    };
                    let Ok(fc) = fc.parse::<usize>() else {
                        return parse_err!("Face count could not be parsed");
                    };
                    f.reserve(fc);
                    num_f = fc;

                    PropertyList
                }
                VertexProperty => {
                    let mut tokens = l.split_whitespace();
                    if tokens.next() != Some("property") {
                        return parse_err!("Missing 'property'");
                    }
                    match tokens.next() {
                        None => return parse_err!("Missing type of property"),
                        Some("uchar") => prop_set.push(Type::UChar),
                        Some("float") => prop_set.push(Type::Float),
                        Some(_) => return parse_err!("Unknown property kind"),
                    }
                    match tokens.next() {
                        None => return parse_err!("Missing property name"),
                        Some("x") => fields.push(Field::X),
                        Some("y") => fields.push(Field::Y),
                        Some("z") => fields.push(Field::Z),

                        Some("red") => fields.push(Field::Red),
                        Some("green") => fields.push(Field::Green),
                        Some("blue") => fields.push(Field::Blue),
                        Some("alpha") => fields.push(Field::Alpha),
                        Some(_) => return parse_err!("Unknown property name"),
                    }
                    let l = fields.last().unwrap();
                    has_color = has_color || matches!(l, Field::Red | Field::Green | Field::Blue);
                    VertexProperty
                }
                PropertyList => {
                    let should_match = ["property", "list", "uchar", "int"];
                    if !l.split_whitespace().take(4).eq(should_match.into_iter()) {
                        return parse_err!(format!("Unknown face property list, got {l:?}, expected {should_match:?}"));
                    }
                    EndHeader
                }
                EndHeader => {
                    if l != "end_header" {
                        return parse_err!("Unknown end of header");
                    }
                    // in theory could match on the beginning of each vertex but lazy
                    if num_v == 0 && num_f == 0 {
                        Done
                    } else if num_v == 0 {
                        Faces
                    } else {
                        Vertices
                    }
                }
                Vertices => {
                    num_v -= 1;

                    let mut xyz = [0.; 3];
                    let mut rgb = [0; 3];

                    for (fi, v) in l.split_whitespace().enumerate() {
                        match fields[fi] {
                            Field::X => xyz[0] = v.parse::<F>().unwrap(),
                            Field::Y => xyz[1] = v.parse::<F>().unwrap(),
                            Field::Z => xyz[2] = v.parse::<F>().unwrap(),
                            Field::Red => rgb[0] = v.parse::<u8>().unwrap(),
                            Field::Green => rgb[1] = v.parse::<u8>().unwrap(),
                            Field::Blue => rgb[2] = v.parse::<u8>().unwrap(),
                            Field::Alpha => continue,
                        }
                    }

                    v.push(xyz);
                    if has_color {
                        vc.push(rgb);
                    }

                    if num_v == 0 {
                        Faces
                    } else {
                        Vertices
                    }
                }
                Faces => {
                    num_f -= 1;

                    use std::array::from_fn;
                    let mut f_tokens = l.split_whitespace();
                    let nf = f_tokens.next().unwrap().parse::<usize>().unwrap();
                    let face = match nf {
                        0 | 1 | 2 => continue,
                        3 => FaceKind::Tri(from_fn(|_| {
                            f_tokens.next().unwrap().parse::<usize>().unwrap()
                        })),
                        4 => FaceKind::Quad(from_fn(|_| {
                            f_tokens.next().unwrap().parse::<usize>().unwrap()
                        })),
                        _ => FaceKind::Poly(
                            f_tokens
                                .map(|t| t.parse::<usize>().unwrap())
                                .collect::<Vec<_>>(),
                        ),
                    };
                    f.push(face);

                    if num_f == 0 {
                        Done
                    } else {
                        Faces
                    }
                }
                Done => {
                    eprintln!("Unexpected extra lines in PLY {l}");
                    Done
                }
            }
        }
        Ok(Ply { v, vc, f })
    }

    /// Write this Ply file to a mesh.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        let has_vc = !self.vc.is_empty();

        writeln!(out, "ply")?;
        writeln!(out, "format ascii 1.0")?;
        writeln!(out, "element vertex {}", self.v.len())?;
        writeln!(out, "property float x")?;
        writeln!(out, "property float y")?;
        writeln!(out, "property float z")?;
        if has_vc {
            assert_eq!(
                self.vc.len(),
                self.v.len(),
                "Mismatch between number of vertices and vertex colors"
            );
            writeln!(out, "property uchar red")?;
            writeln!(out, "property uchar green")?;
            writeln!(out, "property uchar blue")?;
        }

        writeln!(out, "element face {}", self.f.len())?;
        writeln!(out, "property list uchar int vertex_indices")?;
        writeln!(out, "end_header")?;

        for vi in 0..self.v.len() {
            let [x, y, z] = self.v[vi];
            write!(out, "{x} {y} {z}")?;
            if let Some([r, g, b]) = self.vc.get(vi) {
                write!(out, " {r} {g} {b}")?;
            }
            writeln!(out)?;
        }

        for f in &self.f {
            write!(out, "{} ", f.len())?;
            let Some((last, rest)) = f.as_slice().split_last() else {
                continue;
            };
            for vi in rest {
                write!(out, "{vi} ")?;
            }
            writeln!(out, "{last}")?;
        }

        Ok(())
    }
}

#[ignore]
#[test]
fn test_ply_write() {
    use std::fs::{remove_file, File};

    let name = "tmp.ply";
    {
        let f = File::create(name).unwrap();
        let ply = Ply::new(
            vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            vec![FaceKind::Tri([0, 1, 2])],
        );
        ply.write(f).unwrap();
    }
    remove_file(name).expect("Failed to delete file");
}
