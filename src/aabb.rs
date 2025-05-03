use super::{sub, F};
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
        [
            self.min,
            [self.min[0], self.max[1]],
            [self.min[1], self.max[0]],
            self.max,
        ]
    }
    pub fn expand_by(&mut self, v: F) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }
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
