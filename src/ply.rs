use super::Vec3;
use std::io::{self, Write};

#[derive(Debug, Clone, PartialEq)]
pub struct PLY {
    pub vertices: Vec<Vec3>,
    pub colors: Vec<[u8; 3]>,
    pub faces: Vec<[usize; 3]>,
}

impl PLY {
    #[inline]
    pub fn write(&self, mut w: impl Write) -> io::Result<()> {
        writeln!(w, "ply")?;
        writeln!(w, "format ascii 1.0")?;
        writeln!(w, "element vertex {}", self.vertices.len())?;
        writeln!(w, "property float x")?;
        writeln!(w, "property float y")?;
        writeln!(w, "property float z")?;
        assert!(self.colors.is_empty() || self.vertices.len() == self.colors.len());
        if !self.colors.is_empty() {
            // TODO do these need to be uchar?
            writeln!(w, "property uchar red")?;
            writeln!(w, "property uchar green")?;
            writeln!(w, "property uchar blue")?;
        }
        writeln!(w, "element face {}", self.faces.len())?;

        writeln!(w, "property list uchar uint32 vertex_indices")?;
        writeln!(w, "end_header")?;

        for i in 0..self.vertices.len() {
            let v = self.vertices[i];
            write!(w, "{} {} {} ", v[0], v[1], v[2])?;
            if !self.colors.is_empty() {
                let c = self.colors[i];
                write!(w, "{} {} {}", c[0], c[1], c[2])?;
            }
            writeln!(w)?;
        }
        for [i0, i1, i2] in &self.faces {
            writeln!(w, "3 {i0} {i1} {i2}")?;
        }

        Ok(())
    }
}
