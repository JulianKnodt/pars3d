use super::{F, add};

/// Construct a new explicit grid from a given width and height
pub fn new_grid(w: u32, h: u32) -> (Vec<[F; 2]>, Vec<[usize; 4]>) {
    let mut verts = vec![];
    for i in 0..w {
        let x = i as F / w as F;
        for j in 0..h {
            let y = j as F / h as F;
            verts.push([x, y]);
        }
    }

    let mut faces = vec![];
    let idx = |x, y| (x + y * w) as usize;
    for i in 0..w - 1 {
        for j in 0..h - 1 {
            faces.push([idx(i + 1, j), idx(i, j), idx(i, j + 1), idx(i + 1, j + 1)]);
        }
    }
    (verts, faces)
}

pub fn grid_from_delta(
    w: u32,
    h: u32,
    delta: impl Fn([u32; 2]) -> [F; 2],
) -> (Vec<[F; 2]>, Vec<[usize; 4]>) {
    let (mut v, f) = new_grid(w, h);
    for i in 0..w {
        for j in 0..h {
            let tform = |v| v * 2. - 1.;
            let [x, y] = delta([i, j]).map(tform);
            let idx = (i + j * w) as usize;
            v[idx] = add(v[idx], [x, y]);
        }
    }
    (v, f)
}
