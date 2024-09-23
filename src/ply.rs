use std::io::Write;

use super::{FaceKind, F};

/// A PLY mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Ply {
    /// Vertex positions
    v: Vec<[F; 3]>,
    /// Vertex colors
    vc: Vec<[u8; 3]>,

    f: Vec<FaceKind>,
}

impl Ply {
    /// Construct a new PLY mesh, with optional empty vertex colors.
    pub fn new(v: Vec<[F; 3]>, vc: Vec<[u8; 3]>, f: Vec<FaceKind>) -> Self {
        assert!(vc.is_empty() || v.len() == vc.len());
        Self { v, vc, f }
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
