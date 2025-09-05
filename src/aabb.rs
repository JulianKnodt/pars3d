use super::{F, add, kmul, sub};
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
    pub fn center(&self) -> [F; N] {
        add(self.min, kmul(0.5, self.diag()))
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

    /// If this AABB and tri overlap on any axis, returns false.
    pub fn separating_axis_check(&self, tri: [[F; 2]; 3]) -> bool {
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

        true
    }

    pub(crate) fn line_segment_isect(&self, l: [[F; 2]; 2]) -> impl Iterator<Item = [F; 2]> + '_ {
        let c = self.corners();
        (0..4).filter_map(move |i| super::isect::line_segment_isect([c[i], c[(i + 1) % 4]], l))
    }

    /// Returns true if this aabb intersects the tri t.
    pub fn intersects_tri(&self, t: [[F; 2]; 3]) -> bool {
        for c in self.corners() {
            if super::tri_contains(t, c) {
                return true;
            }
        }

        for i in 0..3 {
            let p = t[i];
            if self.contains_point(p) {
                return true;
            }

            for adj in [t[(i + 2) % 3], t[(i + 1) % 3]] {
                if self.line_segment_isect([p, adj]).next().is_some() {
                    return true;
                }
            }
        }

        false
    }

    pub fn intersects_tri_poly(&self, tri: [[F; 2]; 3], poly: &mut Vec<[F; 2]>) {
        #[derive(PartialEq)]
        struct OrdFloat(F);

        impl Eq for OrdFloat {}
        impl Ord for OrdFloat {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                self.0.partial_cmp(&o.0).unwrap()
            }
        }
        impl PartialOrd for OrdFloat {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        poly.clear();

        // check if any points of the aabb are contained in the tri
        poly.extend(
            self.corners()
                .into_iter()
                .filter(|&c| super::tri_contains(tri, c)),
        );

        // for each point on the tri, check if the aabb contains it
        // If it is contained, then it's part of the polygon
        // Otherwise, it is projected to the two adjacent edges

        for i in 0..3 {
            let p = tri[i];
            if self.contains_point(p) {
                poly.push(p);
                continue;
            }
            for adj in [tri[(i + 2) % 3], tri[(i + 1) % 3]] {
                for is in self.line_segment_isect([p, adj]) {
                    poly.push(is);
                }
            }
        }

        // sort polygon counterclockwise
        let center = poly.iter().copied().fold([0.; 2], add);
        let center = kmul((poly.len() as F).recip(), center);

        poly.sort_by_key(|&p| {
            let [x, y] = super::normalize(sub(p, center));
            OrdFloat(y.atan2(x))
        });
        poly.dedup();
    }
}

#[test]
fn test_aabb_isect_tri() {
    let aabb = AABB {
        min: [0.1460009765625, 0.5439501953125],
        max: [0.1464794921875, 0.5444287109375],
    };
    let tri = [
        [0.146492, 0.545388],
        [0.145876, 0.466054],
        [0.146083, 0.466239],
    ];
    assert!(!aabb.intersects_tri(tri));
}

#[test]
fn test_intersection_poly() {
    let mut buf = vec![];
    let tri = [[0., 0.], [1., 0.], [0., 2.]];
    let aabb = AABB {
        min: [0., 0.],
        max: [1., 1.],
    };

    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf, [[0.; 2], [1., 0.], [0.5, 1.], [0., 1.]]);

    let aabb = AABB {
        min: [-1., 0.],
        max: [0., 1.],
    };
    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf.len(), 2);

    let aabb = AABB {
        min: [-10.; 2],
        max: [10.; 2],
    };
    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf, tri);

    let aabb = AABB {
        min: [0., 1.],
        max: [1., 2.],
    };
    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf.len(), 3);

    let aabb = AABB {
        min: [0., 1.],
        max: [1., 2.],
    };
    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf, [[0., 1.], [0.5, 1.], [0., 2.]]);

    let aabb = AABB {
        min: [0.25, 0.25],
        max: [0.4, 0.4],
    };
    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf, aabb.corners());
}

#[test]
fn test_intersection_poly_complex() {
    let mut buf = vec![];
    let tri = [[0., 1.25], [-1.25, 0.], [1.25, 0.]];
    let aabb = AABB {
        min: [-1.; 2],
        max: [1.; 2],
    };

    aabb.intersects_tri_poly(tri, &mut buf);
    assert_eq!(buf.len(), 6, "{buf:?}");
    assert_eq!(
        buf,
        [
            [-1.0, 0.25],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.25],
            [0.25, 1.0],
            [-0.25, 1.0]
        ]
    );
    assert_eq!(1.4375, super::poly_area_2d(&buf));
}
impl AABB<i32, 2> {
    #[inline]
    pub fn width(&self) -> usize {
        (self.max[0] - self.min[0]).max(0) as usize
    }
    #[inline]
    pub fn width_range(&self) -> Range<i32> {
        self.min[0]..self.max[0]
    }
    #[inline]
    pub fn height(&self) -> usize {
        (self.max[1] - self.min[1]).max(0) as usize
    }
    #[inline]
    pub fn height_range(&self) -> Range<i32> {
        self.min[1]..self.max[1]
    }
    #[inline]
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }
    #[inline]
    pub fn iter_coords(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        let [lx, ly] = self.min;
        let [hx, hy] = self.max;
        (ly..hy).flat_map(move |y| (lx..hx).map(move |x| [x, y]))
    }
    #[inline]
    pub fn expand_by(&mut self, v: i32) {
        self.min = self.min.map(|val| val - v);
        self.max = self.max.map(|val| val + v);
    }
    #[inline]
    pub fn intersect(&self, o: &Self) -> Self {
        Self {
            min: from_fn(|i| self.min[i].max(o.min[i])),
            max: from_fn(|i| self.max[i].min(o.max[i])),
        }
    }
}
