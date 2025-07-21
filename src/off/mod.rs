use super::{F, FaceKind};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

pub mod to_mesh;

#[derive(Debug, Default)]
pub struct OFF {
    v: Vec<[F; 3]>,
    f: Vec<FaceKind>,
}

pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<OFF> {
    read(File::open(p.as_ref())?)
}
pub fn read(r: impl Read) -> io::Result<OFF> {
    buf_read(BufReader::new(r))
}

pub fn buf_read(buf_read: impl BufRead) -> io::Result<OFF> {
    let mut off = OFF::default();
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ParseState {
        Init,
        Vert,
        Face,
        Done,
    }

    let mut state = ParseState::Init;
    let mut verts = 0;
    let mut faces = 0;
    let pusize = |s: &str| s.parse::<usize>().unwrap();
    let pf = |s: &str| s.parse::<F>().unwrap();

    for l in buf_read.lines() {
        let l = l?;
        let mut iter = l.split_whitespace();
        let Some(first) = iter.next() else { continue };
        if first.starts_with('#') {
            continue;
        }

        match state {
            ParseState::Done => panic!("Unexpected trailing content {l}"),
            ParseState::Init if first == "OFF" => {}
            ParseState::Init => {
                let Some(face_count) = iter.next() else {
                    panic!("Missing face count in {l}")
                };
                let Some(e) = iter.next() else {
                    panic!("Missing edge count in {l}")
                };
                verts = pusize(first);
                faces = pusize(face_count);
                let _num_edges = pusize(e); // check that it parses

                off.v.reserve(verts);
                off.f.reserve(faces);

                state = ParseState::Vert;
            }
            ParseState::Vert => {
                let Some(y) = iter.next() else {
                    panic!("Missing y in {l}")
                };
                let Some(z) = iter.next() else {
                    panic!("Missing z in {l}")
                };
                off.v.push([first, y, z].map(pf));
            }
            ParseState::Face => {
                let num_verts = pusize(first);
                let mut f = FaceKind::new(num_verts);
                let fs = f.as_mut_slice();
                for i in 0..num_verts {
                    fs[i] = pusize(iter.next().expect("Missing vertex index in OFF face"));
                }
                off.f.push(f);
            }
        }
        if off.v.len() == verts && state == ParseState::Vert {
            state = ParseState::Face;
        }
        if off.f.len() == faces && state == ParseState::Face {
            state = ParseState::Done;
        }
    }

    Ok(off)
}

impl OFF {
    pub fn write(&self, mut dst: impl Write) -> io::Result<()> {
        dst.write_all(b"OFF\n")?;
        writeln!(dst, "{} {} {}", self.v.len(), self.f.len(), 0)?;
        dst.write_all(b"# Vertices:\n")?;
        for [x, y, z] in &self.v {
            writeln!(dst, "{x} {y} {z}")?;
        }
        dst.write_all(b"# Faces:\n")?;
        for f in &self.f {
            let fs = f.as_slice();
            let Some((last, rest)) = fs.split_last() else {
                continue;
            };
            write!(dst, "{} ", fs.len())?;
            for r in rest {
                write!(dst, "{r} ")?;
            }
            writeln!(dst, "{last}")?;
        }
        Ok(())
    }
}
