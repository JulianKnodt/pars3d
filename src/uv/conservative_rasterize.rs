use crate::F;
use crate::aabb::AABB;

use std::num::NonZeroUsize;

/*
fn offset(uv: [F; 2]) -> [i32; 2] {
    uv.map(|s| s.floor() as i32)
}

/// UVs should be normalized to be in the range 0-1, and will be wrapped if outside.
pub fn conservative_rasterize_wrapped(
    uvs: &[[F; 2]],
    tri: [usize; 3],
    output_grid: &mut Grid<bool>,
) {
    let uv_tri = tri.map(|fi| uvs[fi]);
    let os = uv_tri.map(offset);

    // Some offsets are different, need to carefully handle this
    let [[min_ox, min_oy], [max_ox, max_oy]] =
        os.into_iter()
            .fold([[i32::MAX; 2], [i32::MIN; 2]], |[l, h], n| {
                [
                    std::array::from_fn(|i| l[i].min(n[i])),
                    std::array::from_fn(|i| h[i].max(n[i])),
                ]
            });

    match (min_ox == max_ox, min_oy, max_oy) {
        // fast path - all in range (all offsets are same)
        (true, true) => conservative_rasterize(uv_tri.map(|uv| uv.map(F::fract)), output_grid),
        // slow-path (some offsets are different, need to check 2 or 4 tris)
        (false, true) => {
            let uvs0 = uvs.map(|[u, v]| (u - min_ox as F, v - min_oy as F));
            let uvs0 = uvs.map(|[u, v]| (u - min_ox as F, v - min_oy as F));
        }
        (true, false) => {}
        // check 4 tris here
        (false, false) => {
            todo!();
        }
    }
}
*/

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PaddingKind {
    Empty,
    Diamond(NonZeroUsize),
}

impl Default for PaddingKind {
    fn default() -> Self {
        PaddingKind::Diamond(NonZeroUsize::new(1).unwrap())
    }
}

// TODO do different range kinds (diamond, square, circle)
pub fn diamond_cells(
    [u, v]: [usize; 2],
    w: usize,
    h: usize,
    range: usize,
) -> impl Iterator<Item = [usize; 2]> {
    let irange = range as isize;
    ((-irange)..=irange).flat_map(move |x| {
        let rem = (range - x.abs() as usize) as isize;
        ((-rem)..=rem).filter_map(move |y| {
            Some([
                u.checked_sub(x as usize).filter(|&u| u < w)?,
                v.checked_sub(y as usize).filter(|&v| v < h)?,
            ])
        })
    })
}

/// Conservatively rasterizes this triangle into a boolean set.
pub fn conservative_rasterize<T>(
    uv_tri: [[F; 2]; 3],
    w: usize,
    h: usize,
    dst: &mut T,
    get: impl Fn(&T, [usize; 2]) -> bool,
    mut set: impl FnMut(&mut T, [usize; 2]),
) {
    let all_in_range = uv_tri
        .into_iter()
        .all(|uv| uv.into_iter().all(|s| (0.0..1.0).contains(&s)));
    assert!(all_in_range);

    let uv_tri = uv_tri.map(|[u, v]| [u * w as F, v * h as F]);
    let aabb = AABB::from_slice(&uv_tri);
    for [i, j] in aabb.round_to_usize().iter_coords() {
        // Skip cheap case (TODO maybe move this to a parameter to check?)
        if get(dst, [i, j]) {
            continue;
        }
        let i_f = i as F;
        let j_f = j as F;
        let cell_aabb = AABB {
            min: [i_f, j_f],
            max: [i_f + 1., j_f + 1.],
        };
        if !cell_aabb.intersects_tri(uv_tri) {
            continue;
        }
        // mark cell, and adjacent cells
        (&mut set)(dst, [i, j]);
    }
}

#[test]
fn test_conservative_rasterize() {
    use crate::grid::Arr2D;
    const N: usize = 128;
    let mut grid = Arr2D::<bool>::empty(N, N);
    let tri = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25]];
    conservative_rasterize(
        tri,
        N,
        N,
        &mut grid,
        |g, ij| *g.get(ij).unwrap_or(&false),
        |g, ij| *g.get_mut(ij).unwrap() = true,
    );

    use crate::ppm::write as write_ppm;
    use std::fs::File;

    let f = File::create("conservative_rasterize_test.ppm").unwrap();
    write_ppm(f, N, N, |i, j| {
        if *grid.get([i, j]).unwrap() {
            [255; 3]
        } else {
            [0; 3]
        }
    })
    .unwrap();
}
