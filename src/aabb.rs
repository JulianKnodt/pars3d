use super::{add, cross_2d, kmul, sub, F};
use core::ops::Range;
use std::array::from_fn;

/// An axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB<T, const N: usize> {
    pub min: [T; N],
    pub max: [T; N],
}

impl<const N: usize> Default for AABB<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> AABB<F, N> {
    pub fn new() -> Self {
        Self {
            min: [F::INFINITY; N],
            max: [F::NEG_INFINITY; N],
        }
    }
    pub fn add_point(&mut self, p: [F; N]) {
        for i in 0..N {
            self.min[i] = self.min[i].min(p[i]);
            self.max[i] = self.max[i].max(p[i]);
        }
    }
    pub fn diag(&self) -> [F; N] {
        sub(self.max, self.min)
    }
    pub fn from_slice(ps: &[[F; N]]) -> Self {
        let mut s = Self::new();
        for &p in ps {
            s.add_point(p);
        }
        s
    }
    pub fn round_to_i32(&self) -> AABB<i32, N> {
        AABB {
            min: self.min.map(|i| i.floor() as i32),
            max: self.max.map(|i| i.ceil() as i32),
        }
    }
    pub fn scale_by(&mut self, x: F, y: F) {
        self.min[0] *= x;
        self.max[0] *= x;

        self.min[1] *= y;
        self.max[1] *= y;
    }
    #[inline]
    pub fn contains_point(&self, p: [F; N]) -> bool {
        p.iter()
            .enumerate()
            .all(|(dim, &c)| self.within_dim(dim, c))
    }
    #[inline]
    fn within_dim(&self, dim: usize, v: F) -> bool {
        (self.min[dim]..=self.max[dim]).contains(&v)
    }
    pub fn intersection(&self, o: &Self) -> Self {
        Self {
            min: from_fn(|i| self.min[i].max(o.min[i])),
            max: from_fn(|i| self.max[i].min(o.max[i])),
        }
    }
    pub fn union(&self, o: &Self) -> Self {
        Self {
            min: from_fn(|i| self.min[i].min(o.min[i])),
            max: from_fn(|i| self.max[i].max(o.max[i])),
        }
    }
}

impl From<[[F; 2]; 2]> for AABB<F, 2> {
    fn from([a, b]: [[F; 2]; 2]) -> Self {
        AABB {
            min: from_fn(|i| a[i].min(b[i])),
            max: from_fn(|i| a[i].max(b[i])),
        }
    }
}

impl AABB<F, 2> {
    #[inline]
    pub fn corners(&self) -> [[F; 2]; 4] {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        [[lx, ly], [hx, ly], [hx, hy], [lx, hy]]
    }

    #[inline]
    pub fn expand_by(&mut self, v: F) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }

    #[inline]
    pub fn area(&self) -> F {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        (hx - lx) * (hy - ly)
    }

    /// Does this polygon intersect with this aabb? Uses the separating axis theorem
    pub fn intersects_tri(&self, tri: [[F; 2]; 3]) -> bool {
        let [lx, ly] = self.min;
        // quick rejects
        if tri.iter().all(|p| p[0] <= lx) {
            return false;
        }
        if tri.iter().all(|p| p[1] <= ly) {
            return false;
        }
        let [hx, hy] = self.max;
        if tri.iter().all(|p| p[0] >= hx) {
            return false;
        }
        if tri.iter().all(|p| p[1] >= hy) {
            return false;
        }

        if tri.iter().copied().any(|p| self.contains_point(p)) {
            return false;
        }

        // if any edge of the tri intersects with the box, then there is an intersection
        super::edges(&tri).any(|e| self.line_segment_isect(e).next().is_some())
    }

    pub(crate) fn line_segment_isect(&self, l: [[F; 2]; 2]) -> impl Iterator<Item = [F; 2]> + '_ {
        let c = self.corners();
        (0..4).filter_map(move |i| line_segment_isect([c[i], c[(i + 1) % 4]], l))
    }
}

fn line_segment_isect([a0, a1]: [[F; 2]; 2], [b0, b1]: [[F; 2]; 2]) -> Option<[F; 2]> {
    assert_ne!(a0, a1);
    let a_dir = sub(a1, a0);
    assert_ne!(b0, b1);
    let b_dir = sub(b1, b0);

    let denom = cross_2d(a_dir, b_dir);
    if denom == 0. {
        // they're parallel
        return None;
    }
    let t = cross_2d(sub(b0, a0), b_dir) / denom;
    let u = cross_2d(sub(b0, a0), a_dir) / denom;

    let valid = (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u);
    valid.then(|| add(a0, kmul(t, a_dir)))
}

#[test]
fn test_line_segment_isect() {
    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0., 1.], [0., -1.]]);
    assert_eq!(Some([0., 0.]), isect);
    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0., -1.], [0., 1.]]);
    assert_eq!(Some([0., 0.]), isect);

    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0.5, 100.], [-0.5, -100.]]);
    assert_eq!(Some([0., 0.]), isect);
}

impl AABB<i32, 2> {
    pub fn width(&self) -> usize {
        (self.max[0] - self.min[0]).max(0) as usize
    }
    pub fn width_range(&self) -> Range<i32> {
        self.min[0]..self.max[0]
    }
    pub fn height(&self) -> usize {
        (self.max[1] - self.min[1]).max(0) as usize
    }
    pub fn height_range(&self) -> Range<i32> {
        self.min[1]..self.max[1]
    }
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }
    pub fn iter_coords(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        (ly..hy).flat_map(move |y| (lx..hx).map(move |x| [x, y]))
    }
    pub fn expand_by(&mut self, v: i32) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }
    pub fn intersect(&self, o: &Self) -> Self {
        Self {
            min: from_fn(|i| self.min[i].max(o.min[i])),
            max: from_fn(|i| self.max[i].min(o.max[i])),
        }
    }
}
