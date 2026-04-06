use super::{STL, STLFace};
use crate::F;
use std::io::{self, BufRead, BufReader, Read};

use std::path::Path;

impl STL {
    pub fn read_from_file_binary(p: impl AsRef<Path>) -> io::Result<Self> {
        Self::read_binary(std::fs::File::open(p)?)
    }
    pub fn read_binary(r: impl Read) -> io::Result<Self> {
        Self::buf_read_binary(BufReader::new(r))
    }
    pub fn buf_read_binary(mut r: impl BufRead) -> io::Result<Self> {
        let mut header_buf = [0u8; 80];
        r.read_exact(&mut header_buf)?;

        let mut num_tris = [0; 4];
        r.read_exact(&mut num_tris)?;
        let num_tris = u32::from_le_bytes(num_tris);

        macro_rules! read_f32 {
            () => {{
                let mut f = [0; 4];
                r.read_exact(&mut f)?;
                f32::from_le_bytes(f) as F
            }};
        }
        macro_rules! read_u16 {
            () => {{
                let mut u = [0; 2];
                r.read_exact(&mut u)?;
                u16::from_le_bytes(u)
            }};
        }

        let faces = (0..num_tris)
            .map(|_| {
                let normal = [read_f32!(), read_f32!(), read_f32!()];
                let pos: [[F; 3]; 3] = [
                    [read_f32!(), read_f32!(), read_f32!()],
                    [read_f32!(), read_f32!(), read_f32!()],
                    [read_f32!(), read_f32!(), read_f32!()],
                ];
                let _ = read_u16!(/* unused bytes */);
                Ok(STLFace { pos, normal })
            })
            .collect::<io::Result<Vec<_>>>()?;

        Ok(STL {
            name: String::new(),
            faces,
        })
    }
}
