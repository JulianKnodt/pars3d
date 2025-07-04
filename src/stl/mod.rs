use super::{F, Vec3};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

use super::U;

use std::collections::HashMap;

pub mod to_mesh;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct STL {
    name: String,
    faces: Vec<STLFace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct STLFace {
    pos: [Vec3; 3],
    normal: Vec3,
}

pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<STL> {
    read(std::fs::File::open(p)?)
}
pub fn read(reader: impl Read) -> io::Result<STL> {
    buf_read(BufReader::new(reader))
}

pub fn buf_read(reader: impl BufRead) -> io::Result<STL> {
    let mut v = vec![];
    let mut name = String::from("");

    let mut curr_face = STLFace::default();
    let mut curr_vi = 0;

    for l in reader.lines() {
        let line = l?;
        let mut l = line.split_whitespace();

        let parse_float = |s: &str| s.trim().parse::<F>().unwrap();

        let Some(first_token) = l.next() else {
            continue;
        };

        match first_token.trim() {
            // Do nothing first line
            "solid" => {
                if let Some(n) = l.next() {
                    name = String::from(n)
                }
            }
            // last line
            "endsolid" => {}
            "facet" => {
                assert_eq!(l.next(), Some("normal"));
                assert_eq!(curr_vi, 0);
                for i in 0..3 {
                    curr_face.normal[i] = parse_float(l.next().unwrap());
                }
            }
            "vertex" => {
                for i in 0..3 {
                    curr_face.pos[curr_vi][i] = parse_float(l.next().unwrap());
                }
                curr_vi = (curr_vi + 1) % 3;
            }
            "outer" => assert_eq!(l.next(), Some("loop")),
            "endloop" => {}
            "endfacet" => {
                v.push(std::mem::take(&mut curr_face));
            }
            _ => panic!("Unknown line: {line}"),
        }
    }
    Ok(STL { name, faces: v })
}
pub fn write(stl: &STL, mut w: impl Write) -> io::Result<()> {
    writeln!(w, "solid {}", stl.name)?;
    for f in &stl.faces {
        let [nx, ny, nz] = f.normal;
        writeln!(w, "facet normal {nx} {ny} {nz}")?;
        writeln!(w, "outer loop")?;
        for [vx, vy, vz] in f.pos {
            writeln!(w, "vertex {vx} {vy} {vz}")?;
        }
        writeln!(w, "endloop")?;
        writeln!(w, "endfacet")?;
    }
    Ok(())
}

impl STL {
    pub fn stl_raw_to_tri_mesh(&self, merge_distance: F) -> (Vec<Vec3>, Vec<[usize; 3]>) {
        let mut prev_verts = HashMap::new();

        let mut out_verts = vec![];
        let mut out_faces = vec![];

        let vert_key = |xyz: [F; 3]| {
            xyz.map(|v| {
                if merge_distance == 0. {
                    v.to_bits()
                } else {
                    (v / merge_distance) as U
                }
            })
        };

        for face in &self.faces {
            let vis = face.pos.map(|v| {
                *prev_verts.entry(vert_key(v)).or_insert_with(|| {
                    let idx = out_verts.len();
                    out_verts.push(v);
                    idx
                })
            });
            out_faces.push(vis);
        }
        (out_verts, out_faces)
    }
}
