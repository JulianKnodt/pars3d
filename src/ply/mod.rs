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

    /// Vertex Normals
    n: Vec<[F; 3]>,

    /// UV coordinates (called st in PLYs for historical reasons)
    uv: Vec<[F; 2]>,

    /// Faces for this mesh
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

#[derive(PartialEq, Eq, Clone, Copy)]
enum Type {
    UChar,
    Float,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Field {
    X,
    Y,
    Z,
    NX,
    NY,
    NZ,
    S,
    T,
    Red,
    Green,
    Blue,
    Alpha,
    // Allow for storing arbitrary values
    //Arbitrary(String),
}

impl Ply {
    /// Construct a new PLY mesh, with optional empty vertex colors.
    pub fn new(
        v: Vec<[F; 3]>,
        vc: Vec<[u8; 3]>,
        n: Vec<[F; 3]>,
        uv: Vec<[F; 2]>,
        f: Vec<FaceKind>,
    ) -> Self {
        assert!(vc.is_empty() || v.len() == vc.len());
        assert!(n.is_empty() || n.len() == v.len());
        assert!(uv.is_empty() || uv.len() == v.len());
        Self { v, vc, n, uv, f }
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
        let mut has_normal = false;
        let mut has_uv = false;

        let mut v = vec![];
        let mut vc = vec![];
        let mut n = vec![];
        let mut uv: Vec<[F; 2]> = vec![];
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
            if l.trim().starts_with("comment") {
                continue;
            }
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
                    let prop = match tokens.next() {
                        None => return parse_err!("Missing type of property"),
                        Some("uchar") => Type::UChar,
                        Some("float") => Type::Float,
                        Some(_) => return parse_err!("Unknown property kind"),
                    };
                    prop_set.push(prop);

                    let field = match tokens.next() {
                        None => return parse_err!("Missing property name"),
                        Some("x") => Field::X,
                        Some("y") => Field::Y,
                        Some("z") => Field::Z,

                        Some("nx") => Field::NX,
                        Some("ny") => Field::NY,
                        Some("nz") => Field::NZ,

                        Some("s") => Field::S,
                        Some("t") => Field::T,

                        Some("red") => Field::Red,
                        Some("green") => Field::Green,
                        Some("blue") => Field::Blue,
                        Some("alpha") => Field::Alpha,

                        Some(_) => return parse_err!("Unknown property name"),
                    };
                    fields.push(field);
                    has_color =
                        has_color || matches!(field, Field::Red | Field::Green | Field::Blue);
                    has_normal = has_normal || matches!(field, Field::NX | Field::NY | Field::NZ);
                    has_uv = has_uv || matches!(field, Field::S | Field::T);
                    VertexProperty
                }
                PropertyList => {
                    let mut it = l.split_whitespace();
                    let got = [it.next(), it.next(), it.next(), it.next()];
                    let should_match = [Some("property"), Some("list"), Some("uchar"), Some("int")];
                    if got != should_match {
                        return parse_err!(format!(
                            "Unknown face property list, got {got:?}, expected {should_match:?}"
                        ));
                    }
                    use std::assert_matches::assert_matches;
                    assert_matches!(it.next(), Some("vertex_index") | Some("vertex_indices"));
                    EndHeader
                }
                EndHeader => {
                    if l != "end_header" {
                        return parse_err!("Unknown end of header");
                    }
                    // in theory could match on the beginning of each vertex but lazy
                    match (num_v, num_f) {
                        (0, 0) => Done,
                        (0, _) => Faces,
                        _ => Vertices,
                    }
                }
                Vertices => {
                    num_v -= 1;

                    let mut xyz = [0.; 3];
                    let mut rgb = [0; 3];
                    let mut nrm = [0.; 3];
                    let mut uv_ = [0.; 2];

                    for (fi, v) in l.split_whitespace().enumerate() {
                        macro_rules! get {
                            ($t: ty) => {{
                                v.parse::<$t>().unwrap()
                            }};
                        }
                        match fields[fi] {
                            Field::X => xyz[0] = get!(F),
                            Field::Y => xyz[1] = get!(F),
                            Field::Z => xyz[2] = get!(F),

                            Field::NX => nrm[0] = get!(F),
                            Field::NY => nrm[1] = get!(F),
                            Field::NZ => nrm[2] = get!(F),

                            Field::S => uv_[0] = get!(F),
                            Field::T => uv_[1] = get!(F),

                            Field::Red => rgb[0] = get!(u8),
                            Field::Green => rgb[1] = get!(u8),
                            Field::Blue => rgb[2] = get!(u8),
                            Field::Alpha => continue,
                        }
                    }

                    v.push(xyz);
                    if has_color {
                        vc.push(rgb);
                    }
                    if has_normal {
                        n.push(nrm);
                    }
                    if has_uv {
                        uv.push(uv_);
                    }

                    match (num_v, num_f) {
                        (0, 0) => Done,
                        (0, _) => Faces,
                        _ => Vertices,
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
        Ok(Ply { v, vc, uv, n, f })
    }

    /// Write this Ply file to a mesh.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        let has_vc = !self.vc.is_empty();
        let has_n = !self.n.is_empty();
        let has_uv = !self.uv.is_empty();

        writeln!(out, "ply")?;
        writeln!(out, "format ascii 1.0")?;
        writeln!(out, "element vertex {}", self.v.len())?;
        writeln!(out, "property float x")?;
        writeln!(out, "property float y")?;
        writeln!(out, "property float z")?;

        if has_n {
            assert_eq!(
                self.n.len(),
                self.v.len(),
                "Mismatch between #vertices and #normals"
            );
            for p in ["nx", "ny", "nz"] {
                writeln!(out, "property float {p}")?;
            }
        }
        if has_uv {
            assert_eq!(
                self.uv.len(),
                self.v.len(),
                "Mismatch between #uv and #vertices",
            );
            for p in ["s", "t"] {
                writeln!(out, "property float {p}")?;
            }
        }

        if has_vc {
            assert_eq!(
                self.vc.len(),
                self.v.len(),
                "Mismatch between #vertices and #vertex colors"
            );
            for p in ["red", "green", "blue"] {
                writeln!(out, "property uchar {p}")?;
            }
        }

        writeln!(out, "element face {}", self.f.len())?;
        writeln!(out, "property list uchar int vertex_indices")?;
        writeln!(out, "end_header")?;

        for vi in 0..self.v.len() {
            let [x, y, z] = self.v[vi];
            write!(out, "{x} {y} {z}")?;
            if let Some([nx, ny, nz]) = self.n.get(vi) {
                write!(out, " {nx} {ny} {nz}")?;
            }
            if let Some([u, v]) = self.uv.get(vi) {
                write!(out, " {u} {v}")?;
            }
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
            vec![],
            vec![],
            vec![FaceKind::Tri([0, 1, 2])],
        );
        ply.write(f).unwrap();
    }
    remove_file(name).expect("Failed to delete file");
}
