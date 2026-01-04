use super::mesh::{SphHarmonicCoeff, VertexAttrs};

use super::{F, FaceKind};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Error, ErrorKind, Read, Write};
use std::path::Path;

pub mod to_mesh;

/// A PLY mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Ply {
    /// Vertex positions
    pub v: Vec<[F; 3]>,
    /// Vertex colors
    pub vc: Vec<[F; 3]>,

    /// Vertex Normals
    pub n: Vec<[F; 3]>,

    /// UV coordinates (called st in PLYs for historical reasons)
    pub uv: Vec<[F; 2]>,

    /// Faces for this mesh
    pub f: Vec<FaceKind>,

    pub vertex_attrs: VertexAttrs,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
    Height,
    // Allow for storing arbitrary values
    //Arbitrary(String),
    Opacity,

    ScaleX,
    ScaleY,
    ScaleZ,

    RotX,
    RotY,
    RotZ,
    RotW,

    FRest(u8),
}

/// PLY Format
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum FormatKind {
    /// Ascii (Human Readable)
    #[default]
    Ascii,
    /// Binary Little Endian
    BinLil,
    /// Binary Big Endian
    BinBig,
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
        let vc = vc
            .into_iter()
            .map(|v| v.map(|v| v as F / u8::MAX as F))
            .collect::<Vec<_>>();
        Self {
            v,
            vc,
            n,
            uv,
            f,
            vertex_attrs: Default::default(),
        }
    }

    pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<Self> {
        Self::read(File::open(p)?)
    }

    pub fn read(r: impl Read) -> io::Result<Self> {
        Self::buf_read(BufReader::new(r))
    }
    pub fn buf_read(mut r: impl BufRead) -> io::Result<Self> {
        let mut state = ReadExpect::Header;
        let mut prop_set = vec![];
        let mut fields = vec![];

        let mut has_color = false;
        let mut has_normal = false;
        let mut has_uv = false;
        let mut has_height = false;
        let mut has_opacity = false;
        let mut has_scale = false;
        let mut has_rot = false;
        let mut has_f_rest = false;

        let mut v = vec![];
        let mut vc: Vec<[F; 3]> = vec![];
        let mut n = vec![];
        let mut uv: Vec<[F; 2]> = vec![];
        let mut vertex_attrs = VertexAttrs::default();

        let opacitys = &mut vertex_attrs.opacity;
        let height = &mut vertex_attrs.height;
        let scales = &mut vertex_attrs.scale;
        let rots = &mut vertex_attrs.rot;
        let sphs = &mut vertex_attrs.sph_harmonic_coeff;

        let mut f = vec![];

        let mut num_v = 0;
        let mut num_f = 0;

        let mut vertex_bytes = 0;

        macro_rules! parse_err {
            ($s: expr) => {
                Err(Error::new(ErrorKind::Other, $s))
            };
        }

        let mut format = FormatKind::Ascii;

        let mut buf = vec![];
        loop {
            buf.clear();
            match format {
                FormatKind::Ascii => {
                    if r.read_until(b'\n', &mut buf)? == 0 {
                        break;
                    }
                }
                FormatKind::BinLil | FormatKind::BinBig => match state {
                    ReadExpect::Done => break,
                    ReadExpect::Vertices => {
                        buf.resize(vertex_bytes, 0);
                        r.read_exact(&mut buf)?;
                    }
                    ReadExpect::Faces => todo!(),
                    _ => {
                        if r.read_until(b'\n', &mut buf)? == 0 {
                            break;
                        }
                    }
                },
            }
            let l = buf
                .strip_suffix(b"\n")
                .and_then(|b| std::str::from_utf8(b).ok());
            if l.is_some_and(|l| l.trim().starts_with("comment")) {
                continue;
            }
            use ReadExpect::*;
            state = match state {
                Header => {
                    if l != Some("ply") {
                        return parse_err!(format!("Expected ply as 1st line, got {l:?}"));
                    } else {
                        Format
                    }
                }
                Format => {
                    format = match l.expect("Format should be ascii") {
                        "format ascii 1.0" => FormatKind::Ascii,
                        "format binary_little_endian 1.0" => FormatKind::BinLil,
                        "format binary_big_endian 1.0" => FormatKind::BinBig,
                        _ => return parse_err!(format!("Unsupported ply format (got {l:?})")),
                    };
                    VertexCount
                }
                VertexCount => {
                    let mut tokens = l.unwrap().split_whitespace();
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
                VertexProperty if l == Some("end_header") => {
                    // in theory could match on the beginning of each vertex but lazy
                    match (num_v, num_f) {
                        (0, 0) => Done,
                        (0, _) => Faces,
                        _ => Vertices,
                    }
                }
                VertexProperty
                    if let Some(l) = l
                        && l.starts_with("element") =>
                {
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
                    let mut tokens = l.unwrap().split_whitespace();
                    if tokens.next() != Some("property") {
                        return parse_err!(format!("Missing 'property', got {:?} instead", l));
                    }
                    let prop = match tokens.next() {
                        None => return parse_err!("Missing type of property"),
                        Some("uchar") => Type::UChar,
                        Some("float") => Type::Float,
                        Some(_) => return parse_err!("Unknown property kind"),
                    };
                    vertex_bytes += match prop {
                        Type::UChar => 1,
                        Type::Float => 4,
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
                        Some("height") => Field::Height,

                        // Gaussian splatting parameters
                        Some(k) if k.starts_with("f_dc_") => {
                            match k["f_dc_".len()..].parse().unwrap() {
                                0 => Field::Red,
                                1 => Field::Green,
                                2 => Field::Blue,
                                _ => return parse_err!(format!("Unknown property {k}")),
                            }
                        }
                        Some(k) if k.starts_with("f_rest_") => {
                            Field::FRest(k["f_rest_".len()..].parse().unwrap())
                        }
                        Some("opacity") => Field::Opacity,

                        Some("scale_0") => Field::ScaleX,
                        Some("scale_1") => Field::ScaleY,
                        Some("scale_2") => Field::ScaleZ,

                        Some("rot_0") => Field::RotX,
                        Some("rot_1") => Field::RotY,
                        Some("rot_2") => Field::RotZ,
                        Some("rot_3") => Field::RotW,

                        Some(k) => return parse_err!(format!("Unknown property name {k}")),
                    };
                    fields.push(field);
                    use Field::*;
                    has_color = has_color || matches!(field, Red | Green | Blue);
                    has_normal = has_normal || matches!(field, NX | NY | NZ);
                    has_uv = has_uv || matches!(field, S | T);
                    has_height = has_height || matches!(field, Height);
                    has_opacity = has_opacity || matches!(field, Opacity);
                    has_scale = has_scale || matches!(field, ScaleX | ScaleY | ScaleZ);
                    has_rot = has_rot || matches!(field, RotX | RotY | RotZ | RotW);
                    has_f_rest = has_f_rest || matches!(field, FRest(_));
                    VertexProperty
                }
                PropertyList => {
                    let mut it = l.unwrap().split_whitespace();
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
                    if l != Some("end_header") {
                        return parse_err!("Unknown end of header {l:?}");
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
                    let mut rgb = [0.; 3];
                    let mut nrm = [0.; 3];
                    let mut uv_ = [0.; 2];
                    let mut h = 0.;
                    let mut opacity = 0.;

                    let mut scale = [0.; 3];
                    let mut rot = [0.; 4];
                    let mut sph = [SphHarmonicCoeff::default(); 3];

                    let mut offset = 0;
                    match format {
                        FormatKind::BinLil | FormatKind::BinBig => {
                            macro_rules! get {
                                ($raw_t: ty, $t: ty) => {{
                                    use std::array::from_fn;
                                    let v = match format {
                                        FormatKind::Ascii => unreachable!(),
                                        FormatKind::BinLil => {
                                            <$raw_t>::from_le_bytes(from_fn(|i| buf[i + offset]))
                                        }
                                        FormatKind::BinBig => {
                                            <$raw_t>::from_be_bytes(from_fn(|i| buf[i + offset]))
                                        }
                                    };
                                    offset += std::mem::size_of::<$raw_t>();
                                    v as $t
                                }};
                            }

                            for (i, field) in fields.iter().enumerate() {
                                macro_rules! cond_parse {
                                    () => {{
                                        match prop_set[i] {
                                            Type::UChar => get!(u8, u8) as F / u8::MAX as F,
                                            Type::Float => get!(f32, F),
                                        }
                                    }};
                                }
                                match field {
                                    Field::X => xyz[0] = get!(f32, F),
                                    Field::Y => xyz[1] = get!(f32, F),
                                    Field::Z => xyz[2] = get!(f32, F),

                                    Field::NX => nrm[0] = get!(f32, F),
                                    Field::NY => nrm[1] = get!(f32, F),
                                    Field::NZ => nrm[2] = get!(f32, F),

                                    Field::S => uv_[0] = get!(f32, F),
                                    Field::T => uv_[1] = get!(f32, F),

                                    Field::Red => rgb[0] = cond_parse!(),
                                    Field::Green => rgb[1] = cond_parse!(),
                                    Field::Blue => rgb[2] = cond_parse!(),
                                    Field::Height => h = get!(f32, F),
                                    Field::Alpha => todo!(),

                                    Field::Opacity => opacity = get!(f32, F),
                                    Field::FRest(n) => {
                                        let n = *n as usize;
                                        // maybe this is incorrect (but it's the most simple)
                                        let color = n / 15;
                                        let chan = n % 15;
                                        /*
                                        let color = n % 3;
                                        let chan = n / 3;
                                        assert!(chan < 15);
                                        */
                                        *sph[color].get_mut(chan).unwrap() = get!(f32, F);
                                    }

                                    Field::ScaleX => scale[0] = get!(f32, F),
                                    Field::ScaleY => scale[1] = get!(f32, F),
                                    Field::ScaleZ => scale[2] = get!(f32, F),

                                    Field::RotX => rot[0] = get!(f32, F),
                                    Field::RotY => rot[1] = get!(f32, F),
                                    Field::RotZ => rot[2] = get!(f32, F),
                                    Field::RotW => rot[3] = get!(f32, F),
                                }
                            }
                        }
                        FormatKind::Ascii => {
                            for (fi, v) in l.unwrap().split_whitespace().enumerate() {
                                macro_rules! get {
                                    ($t: ty) => {{ v.parse::<$t>().unwrap() }};
                                }
                                macro_rules! cond_parse {
                                    () => {{
                                        match prop_set[fi] {
                                            Type::UChar => get!(u8) as F / u8::MAX as F,
                                            Type::Float => get!(F),
                                        }
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

                                    Field::Red => rgb[0] = cond_parse!(),
                                    Field::Green => rgb[1] = cond_parse!(),
                                    Field::Blue => rgb[2] = cond_parse!(),
                                    Field::Height => h = get!(F),
                                    Field::Alpha => continue,

                                    Field::Opacity => opacity = get!(F),
                                    f => todo!("{f:?}"),
                                }
                            }
                        }
                    }

                    v.push(xyz);
                    macro_rules! cond_push {
                        ($cond: expr, $dest: ident, $val: ident) => {{
                            if $cond {
                                $dest.push($val);
                            }
                        }};
                    }

                    cond_push!(has_color, vc, rgb);
                    cond_push!(has_normal, n, nrm);
                    cond_push!(has_uv, uv, uv_);
                    cond_push!(has_height, height, h);
                    cond_push!(has_opacity, opacitys, opacity);
                    cond_push!(has_scale, scales, scale);
                    cond_push!(has_rot, rots, rot);
                    cond_push!(has_f_rest, sphs, sph);

                    match (num_v, num_f) {
                        (0, 0) => Done,
                        (0, _) => Faces,
                        _ => Vertices,
                    }
                }
                Faces => {
                    assert_eq!(format, FormatKind::Ascii, "TODO implement faces in binary");
                    num_f -= 1;

                    use std::array::from_fn;
                    let mut f_tokens = l.unwrap().split_whitespace();
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

                    if num_f == 0 { Done } else { Faces }
                }
                Done => {
                    if let Some(l) = l {
                        eprintln!("Unexpected extra lines in PLY {l:?}");
                    }
                    break;
                }
            }
        }

        Ok(Ply {
            v,
            vc,
            uv,
            n,
            f,
            vertex_attrs,
        })
    }

    /// Write this Ply file to a mesh.
    pub fn write(&self, mut out: impl Write, fmt: FormatKind) -> std::io::Result<()> {
        let has_vc = !self.vc.is_empty();
        let has_n = !self.n.is_empty();
        let has_uv = !self.uv.is_empty();
        let va = &self.vertex_attrs;

        let has_h = !va.height.is_empty();
        // gaussian splat attributes
        let has_opacity = !va.opacity.is_empty();
        let has_scale = !va.scale.is_empty();
        let has_rot = !va.rot.is_empty();
        let has_sph = !va.sph_harmonic_coeff.is_empty();

        writeln!(out, "ply")?;
        match fmt {
            FormatKind::Ascii => writeln!(out, "format ascii 1.0")?,
            FormatKind::BinBig => writeln!(out, "format binary_big_endian 1.0")?,
            FormatKind::BinLil => writeln!(out, "format binary_little_endian 1.0")?,
        }
        writeln!(out, "element vertex {}", self.v.len())?;
        writeln!(out, "property float x")?;
        writeln!(out, "property float y")?;
        writeln!(out, "property float z")?;

        macro_rules! opt_property {
            ($cond: expr, $src: expr, $name: literal, $ty_name: expr, $exprs: expr) => {{
                if $cond {
                    assert_eq!(
                        self.v.len(),
                        $src.len(),
                        "Mismatch between #vertices and #{}",
                        $name
                    );
                    for p in $exprs {
                        writeln!(out, "property {} {p}", $ty_name)?;
                    }
                }
            }};
        }

        opt_property!(has_n, self.n, "normals", "float", ["nx", "ny", "nz"]);
        opt_property!(has_uv, self.uv, "uv", "float", ["s", "t"]);
        if has_sph {
            let attrs = ["f_dc_0", "f_dc_1", "f_dc_2"];
            opt_property!(has_vc, self.vc, "vertex colors", "float", attrs);
        } else {
            let attrs = ["red", "green", "blue"];
            opt_property!(has_vc, self.vc, "vertex colors", "uchar", attrs);
        }

        let attrs = (0..45).map(|i| format!("f_rest_{i}"));
        opt_property!(has_sph, va.sph_harmonic_coeff, "sphs", "float", attrs);

        opt_property!(has_h, va.height, "height", "float", ["height"]);
        opt_property!(has_opacity, va.opacity, "opacity", "float", ["opacity"]);
        let attrs = ["scale_0", "scale_1", "scale_2"];
        opt_property!(has_scale, va.scale, "scale", "float", attrs);
        let attrs = ["rot_0", "rot_1", "rot_2", "rot_3"];
        opt_property!(has_rot, va.rot, "rot", "float", attrs);

        if !(self.f.is_empty() && has_sph) {
            writeln!(out, "element face {}", self.f.len())?;
            writeln!(out, "property list uchar int vertex_indices")?;
        }
        writeln!(out, "end_header")?;

        macro_rules! write_out {
            (OPT: $data: expr) => {{
                if let Some(d) = $data {
                    write_out!(d);
                }
            }};
            ($data: expr) => {{
                for v in $data {
                    match fmt {
                        FormatKind::Ascii => write!(out, " {v}")?,
                        FormatKind::BinLil => {
                            out.write_all(&v.to_le_bytes())?;
                        }
                        FormatKind::BinBig => {
                            out.write_all(&v.to_be_bytes())?;
                        }
                    }
                }
            }};
            ($data: expr, NO_LEADING_SPACE) => {{
                let mut d = $data.into_iter();
                if let Some(v) = d.next() {
                    match fmt {
                        FormatKind::Ascii => write!(out, "{v}")?,
                        FormatKind::BinLil => {
                            out.write_all(&v.to_le_bytes())?;
                        }
                        FormatKind::BinBig => {
                            out.write_all(&v.to_be_bytes())?;
                        }
                    }
                }
                write_out!(d);
            }};
        }
        for vi in 0..self.v.len() {
            write_out!(self.v[vi], NO_LEADING_SPACE);
            write_out!(OPT: self.n.get(vi));
            write_out!(OPT: self.uv.get(vi));
            if let Some(rgb) = self.vc.get(vi) {
                if has_sph {
                    write_out!(rgb);
                } else {
                    let rgb = rgb.map(|v| (v.clamp(0., 1.) * u8::MAX as F) as u8);
                    write_out!(rgb);
                }
            }
            if let Some(sphs) = va.sph_harmonic_coeff.get(vi) {
                let coeffs = sphs
                    .iter()
                    .flat_map(|sph| (0..15).map(|i| sph.get(i).unwrap()));
                write_out!(coeffs);
            }
            write_out!(OPT: va.height.get(vi).map(|h| [h]));
            write_out!(OPT: va.opacity.get(vi).map(|o| [o]));
            write_out!(OPT: va.scale.get(vi));
            write_out!(OPT: va.rot.get(vi));
            if let FormatKind::Ascii = fmt {
                writeln!(out)?;
            }
        }

        for f in &self.f {
            write_out!([f.len() as u8], NO_LEADING_SPACE);
            write_out!(f.as_slice().iter().map(|&i| i as i32));
            if let FormatKind::Ascii = fmt {
                writeln!(out)?;
            }
        }

        Ok(())
    }
}

#[ignore]
#[test]
fn test_ply_write() {
    use std::fs::{File, remove_file};

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
        ply.write(f, Default::default()).unwrap();
    }
    remove_file(name).expect("Failed to delete file");
}
