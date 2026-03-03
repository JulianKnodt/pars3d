use std::io::{self, Write};

/// Write a grid to a file in PPM format.
pub fn write(
    mut dst: impl Write,
    w: usize,
    h: usize,
    c: impl Fn(usize, usize) -> [u8; 3],
) -> io::Result<()> {
    writeln!(dst, "P3")?;
    writeln!(dst, "{w} {h}")?;
    writeln!(dst, "255")?;

    for i in 0..w {
        for j in 0..h {
            let [r, g, b] = c(i, j);
            writeln!(dst, "{r} {g} {b} ")?;
        }
    }

    Ok(())
}
