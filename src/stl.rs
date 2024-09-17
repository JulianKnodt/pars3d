use super::{into, Vec3, F};
use std::io::{self, BufRead, BufReader, Read};

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct STLFace {
    pos: [Vec3; 3],
    normal: Vec3,
}

pub fn read(reader: impl Read) -> io::Result<Vec<STLFace>> {
    let mut v = vec![];

    let mut curr_face = STLFace::default();
    let mut curr_vi = 0;

    let reader = BufReader::new(reader);

    for l in reader.lines() {
        let line = l?;
        let mut l = line.split_whitespace();

        let parse_float = |s: &str| s.trim().parse::<F>().unwrap();

        let Some(first_token) = l.next() else {
            continue;
        };

        match first_token.trim() {
            // Do nothing first line
            "solid" => {}
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
    Ok(v)
}

pub fn stl_raw_to_tri_mesh(v: &[STLFace], merge_distance: F) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let mut prev_verts = HashMap::new();

    let mut out_verts = vec![];
    let mut out_faces = vec![];

    let vert_key = |xyz: [F; 3]| xyz.map(|v| into(v / merge_distance) as u32);

    for face in v {
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
