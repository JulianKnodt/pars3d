#![allow(unused)]

pub mod to_mesh;

use super::{FaceKind, F};

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// Represents the geometry in a VMRL file.
#[derive(Default)]
pub struct VRML {
    groups: Vec<Group>,
}

#[derive(Default)]
pub struct Group {
    children: Vec<Child>,
}

#[derive(Default)]
pub struct Child {
    shape: Shape,
}

#[derive(Default)]
pub struct Shape {
    ccw: bool,
    solid: bool,
    convex: bool,
    pub points: Vec<[F; 3]>,
    pub indices: Vec<FaceKind>,
}

/*
impl VRML {
  pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<Self> {
    Self::read(File::open(p)?)
  }
  pub fn read(r: impl Read) -> io::Result<Self> {
    Self::buf_read(BufReader::new(r))
  }
  pub fn buf_read(r: impl BufRead) -> io::Result<Self> {
    let mut out = VRML::default();
    let mut parse_state = ParseState::default()
    for l in r.lines() {
      let l = l?;
      if l.trim().starts_with("#") || l.trim().is_empty() {
        continue;
      }
      todo!();
    }
  }
}
*/

#[derive(Default)]
pub struct VRMLGeometryOnly {
    pub shapes: Vec<Shape>,
}

impl VRMLGeometryOnly {
    pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<Self> {
        Self::read(File::open(p)?)
    }
    pub fn read(r: impl Read) -> io::Result<Self> {
        Self::buf_read(BufReader::new(r))
    }
    pub fn buf_read(r: impl BufRead) -> io::Result<Self> {
        let mut out = VRMLGeometryOnly::default();
        let mut curr_shape = Shape::default();
        let mut prev_was_float = true;
        for l in r.lines() {
            let l = l?;
            if l.trim().starts_with("#") || l.trim().is_empty() {
                continue;
            }
            let mut tokens = l.split_whitespace();
            let Some(t0) = tokens.next() else {
                continue;
            };
            let Some(t1) = tokens.next() else {
                continue;
            };
            let Some(t2) = tokens.next() else {
                continue;
            };
            let is_float = t0.contains(".") || t1.contains(".") || t2.contains(".");
            if is_float {
                if !prev_was_float {
                    out.shapes.push(std::mem::take(&mut curr_shape));
                    prev_was_float = true;
                }
                let p = [t0, t1, t2].map(|v| v.parse::<F>().unwrap());
                curr_shape.points.push(p);
            } else {
                prev_was_float = false;
                let p = [t0, t1, t2].map(|v| v.parse::<usize>().unwrap());
                curr_shape.indices.push(FaceKind::Tri(p));
            }
        }

        if !curr_shape.points.is_empty() {
            out.shapes.push(curr_shape);
        }
        Ok(out)
    }

    pub fn to_vrml(self) -> VRML {
        let groups = self
            .shapes
            .into_iter()
            .map(|shape| Group {
                children: vec![Child { shape }],
            })
            .collect();
        VRML { groups }
    }
}
