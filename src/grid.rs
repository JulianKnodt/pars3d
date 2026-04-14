use super::{F, add};
use crate::aabb::AABB;

/// Construct a new explicit grid from a given width and height
/// Returns 2D coordinates, along side quads that represent each grid cell.
pub fn new_grid(w: u32, h: u32) -> (Vec<[F; 2]>, Vec<[usize; 4]>) {
    let mut verts = vec![];
    for j in 0..h {
        let y = j as F / h as F;
        for i in 0..w {
            let x = i as F / w as F;
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

// --- An actual grid structure

// Struct that represents a 2D array
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arr2D<T, const ROW_MAJOR: bool = true> {
    pub h: usize,
    pub w: usize,
    pub(crate) data: Vec<T>,
}

impl<T, const ORD: bool> Arr2D<T, ORD> {
    #[inline]
    pub fn empty(w: usize, h: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            h,
            w,
            data: vec![T::default(); h * w],
        }
    }
    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        [self.w, self.h]
    }

    /// Fills all elements of this array with a new value.
    #[inline]
    pub fn fill(&mut self, v: T)
    where
        T: Copy,
    {
        self.data.fill(v)
    }

    /// Fills all elements of this array with a new value.
    #[inline]
    pub fn fill_with(&mut self, cons: impl Fn() -> T) {
        self.data.fill_with(cons)
    }
    /// Unspecified order of iteration.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Unspecified order of iteration.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn resize(&mut self, w: usize, h: usize, v: T)
    where
        T: Clone,
    {
        self.w = w;
        self.h = h;
        self.data.resize(w * h, v);
    }
    #[inline]
    pub fn truncate(&mut self, w: usize, h: usize) {
        assert!(self.data.len() >= w * h);
        self.w = w;
        self.h = h;
        self.data.truncate(w * h);
    }
    #[inline]
    pub fn resize_with(&mut self, w: usize, h: usize, v: impl Fn() -> T) {
        self.w = w;
        self.h = h;
        self.data.resize_with(w * h, v);
    }

    #[inline]
    fn idx(&self, [x, y]: [usize; 2]) -> usize {
        if ORD { x + y * self.w } else { y + x * self.h }
    }
    #[inline]
    pub fn get(&self, xy: [usize; 2]) -> Option<&T> {
        self.data.get(self.idx(xy))
    }

    /*
    #[inline]
    fn get_idx(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }
    */
    #[inline]
    pub fn get_mut(&mut self, xy: [usize; 2]) -> Option<&mut T> {
        let idx = self.idx(xy);
        self.data.get_mut(idx)
    }

    pub fn iter_enumerate(&self) -> impl Iterator<Item = ([usize; 2], &T)> {
        self.data.iter().enumerate().map(|(i, v)| {
            let xy = if ORD {
                [i % self.w, i / self.w]
            } else {
                [i / self.h, i % self.h]
            };
            (xy, v)
        })
    }
    pub fn iter_enumerate_region(
        &self,
        aabb: &AABB<usize, 2>,
    ) -> impl Iterator<Item = ([usize; 2], &T)> {
        self.data.iter().enumerate().filter_map(move |(i, v)| {
            let xy = if ORD {
                [i % self.w, i / self.w]
            } else {
                [i / self.h, i % self.h]
            };
            aabb.contains(xy).then_some((xy, v))
        })
    }

    pub fn iter_mut_enumerate(&mut self) -> impl Iterator<Item = ([usize; 2], &mut T)> {
        self.data.iter_mut().enumerate().map(|(i, v)| {
            let xy = if ORD {
                [i % self.w, i / self.w]
            } else {
                [i / self.h, i % self.h]
            };
            (xy, v)
        })
    }

    pub fn region_enumerate(
        &self,
        [lx, ly]: [usize; 2],
        [hx, hy]: [usize; 2],
    ) -> impl Iterator<Item = ([usize; 2], &T)> {
        let l = [lx.min(self.w), ly.min(self.h)];
        let h = [hx.min(self.w), hy.min(self.h)];

        let i0 = self.idx(l);
        let i1 = self.idx(h);
        self.data
            .iter()
            .enumerate()
            .skip(i0)
            .take(i1 - i0 + 1)
            .filter_map(move |(i, v)| {
                let xy = if ORD {
                    [i % self.w, i / self.w]
                } else {
                    [i / self.h, i % self.h]
                };
                if !(l[0]..=h[0]).contains(&xy[0]) {
                    return None;
                }
                if !(l[1]..=h[1]).contains(&xy[1]) {
                    return None;
                }
                Some((xy, v))
            })
    }
    /*
    fn adj_rows(&mut self, y: usize) -> (&mut [T], &[T]) {
        assert!(ORD);
        let two_rows = &mut self.data[y * self.w..self.w * (y + 2)];
        let (a, b) = two_rows.split_at_mut(self.w);
        (a, b)
    }
    fn adj_cols(&mut self, x: usize) -> (&mut [T], &[T]) {
        assert!(!ORD);
        let two_rows = &mut self.data[x * self.h..self.h * (x + 2)];
        let (a, b) = two_rows.split_at_mut(self.h);
        (a, b)
    }
    */

    pub fn elemwise_op_assign(&mut self, o: &Self, op: impl Fn(&T, &T) -> T)
    where
        T: Copy,
    {
        for (l, r) in self.iter_mut().zip(o.iter()) {
            *l = op(&l, &r);
        }
    }
}

/*
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        // [a,b] = [b, a % b]
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
*/

impl<T> Arr2D<T, true> {
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.w..y * (self.w + 1)]
    }
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[y * self.w..self.w * (y + 1)]
    }

    #[inline]
    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> T + Copy) -> Self {
        Self {
            h,
            w,
            data: (0..w)
                .flat_map(|x| (0..h).map(move |y| f(x, y)))
                .collect::<Vec<T>>(),
        }
    }
}

impl<T> Arr2D<T, false> {
    pub fn col(&self, x: usize) -> &[T] {
        &self.data[x * self.h..x * (self.h + 1)]
    }
    pub fn col_mut(&mut self, x: usize) -> &mut [T] {
        &mut self.data[x * self.h..self.h * (x + 1)]
    }
    #[inline]
    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> T + Copy) -> Self {
        Self {
            h,
            w,
            data: (0..h)
                .flat_map(|y| (0..w).map(move |x| f(x, y)))
                .collect::<Vec<T>>(),
        }
    }
}

impl<T> Arr2D<Vec<T>> {
    #[inline]
    pub fn clear(&mut self)
    where
        T: Send + Sync,
    {
        self.data.iter_mut().for_each(Vec::clear);
    }
}

impl<const ORD: bool> Arr2D<bool, ORD> {
    pub fn any(&self) -> bool {
        self.data.iter().any(|&v| v)
    }
    /// Returns the region where any value is true
    pub fn true_aabb(&self) -> AABB<usize, 2> {
        let mut out = AABB::new_usize();
        for (ij, v) in self.iter_enumerate() {
            if !v {
                continue;
            }
            out.add_point(ij);
        }
        return out;
    }
    pub fn union<const OORD: bool>(&mut self, o: &Arr2D<bool, OORD>, [ox, oy]: [usize; 2]) {
        for ([x, y], &src) in o.iter_enumerate() {
            let Some(dst) = self.get_mut([x + ox, y + oy]) else {
                continue;
            };
            *dst = src;
        }
    }

    /// Convolves two boolean signals.
    pub fn convolve<const ORD1: bool, const ORD2: bool>(
        &self,
        kernel: &Arr2D<bool, ORD1>,
        dst: &mut Arr2D<bool, ORD2>,
    ) {
        assert_eq!(self.shape(), dst.shape());
        dst.fill(false);
        if !kernel.any() || !self.any() {
            return;
        }

        let true_aabb = kernel.true_aabb();
        // for each true, mark region in surrounding area as true
        // OR for each cell check nearby region?

        for ([i, j], _) in self.iter_enumerate() {
            let d = dst.get_mut([i, j]).unwrap();
            if *d {
                continue;
            }
            for ([ki, kj], ov) in kernel.iter_enumerate_region(&true_aabb) {
                if !ov {
                    continue;
                }
                let Some(sv) = self.get([i + ki, j + kj]) else {
                    continue;
                };
                if !sv {
                    continue;
                }
                *d = true;
                break;
            }
        }
    }

    /// Convolves two boolean signals, where if either contains `true` in range
    /// the output will have true.
    pub fn minkowski_sum<const ORD1: bool, const ORD2: bool>(
        &self,
        kernel: &Arr2D<bool, ORD1>,
        dst: &mut Arr2D<bool, ORD2>,
    ) {
        assert_eq!(self.shape(), dst.shape());
        dst.fill(false);
        if !kernel.any() || !self.any() {
            return;
        }

        // TODO how to handle even-size kernel?
        let true_aabb = kernel.true_aabb();
        // for each true, mark region in surrounding area as true
        // OR for each cell check nearby region?

        for ([i, j], v) in self.iter_enumerate() {
            if !v {
                continue;
            }
            let d = dst.get_mut([i, j]).unwrap();
            if *d {
                continue;
            }
            for ([ki, kj], ov) in kernel.iter_enumerate_region(&true_aabb) {
                if !ov {
                    continue;
                }
                let Some(sv) = self.get([i + ki, j + kj]) else {
                    continue;
                };
                if !sv {
                    continue;
                }
                *d = true;
                break;
            }
        }
    }
}

#[test]
fn test_bool_convolve() {
    const N: usize = 32;
    let mut a = Arr2D::<bool>::empty(N, N);
    a.row_mut(16)[12..20].fill(true);
    let mut b = Arr2D::<bool>::empty(3, 5);
    *b.get_mut([1, 0]).unwrap() = true;
    *b.get_mut([1, 1]).unwrap() = true;
    *b.get_mut([1, 2]).unwrap() = true;
    *b.get_mut([0, 2]).unwrap() = true;
    *b.get_mut([2, 2]).unwrap() = true;
    *b.get_mut([1, 3]).unwrap() = true;
    *b.get_mut([1, 4]).unwrap() = true;

    let mut dst = Arr2D::<bool>::empty(N, N);
    a.convolve(&b, &mut dst);

    for i in 0..N {
        for j in 0..N {
            if *dst.get([i, j]).unwrap() {
                print!("x ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
    todo!();
}
