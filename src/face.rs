use super::aabb::AABB;
use crate::{
    F, add, barycentric_2d, barycentric_3d, cross, cross_2d, dot, edges, kmul, length, sub,
};

/// Face representation for meshes.
/// Tris and quads are stack allocated,
/// If you're a madman and store general polygons they're on the heap.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceKind<T = usize> {
    Tri([T; 3]),
    Quad([T; 4]),
    Poly(Vec<T>),
}

// TODO shrink this to 32 or smaller
const _: () = assert!(
    std::mem::size_of::<FaceKind<usize>>() == 40,
    "Size of FaceKind is too large",
);

// TODO shrink this to 16 or smaller
const _: () = assert!(
    std::mem::size_of::<FaceKind<u32>>() == 24,
    "Size of FaceKind is too large",
);

impl<T> FaceKind<T> {
    pub fn as_slice(&self) -> &[T] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_slice(),
            Quad(q) => q.as_slice(),
            Poly(v) => v.as_slice(),
        }
    }
    pub fn num_tris(&self) -> usize {
        self.as_slice().len().saturating_sub(2)
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_mut_slice(),
            Quad(q) => q.as_mut_slice(),
            Poly(v) => v.as_mut_slice(),
        }
    }
    /// Number of vertices in this face.
    pub fn len(&self) -> usize {
        use FaceKind::*;
        match self {
            Tri(_) => 3,
            Quad(_) => 4,
            Poly(v) => v.len(),
        }
    }
    /// `true` if there are any vertices in this face.
    pub fn is_empty(&self) -> bool {
        use FaceKind::*;
        match self {
            Tri(_) => false,
            Quad(_) => false,
            Poly(v) => v.is_empty(),
        }
    }

    /// Returns an empty face. Can be used as a tombstone, or during construction of a face.
    pub fn empty() -> Self {
        FaceKind::Poly(vec![])
    }

    #[inline]
    pub fn is_tri(&self) -> bool {
        matches!(self, FaceKind::Tri(_))
    }
    #[inline]
    pub fn is_quad(&self) -> bool {
        matches!(self, FaceKind::Quad(_))
    }

    /// Returns each edge in this face: [vi0, vi1], [vi1, vi2]... [viN, vi0]
    pub fn edges(&self) -> impl Iterator<Item = [T; 2]> + '_
    where
        T: Copy,
    {
        edges(self.as_slice())
    }

    /// Returns [prev, curr, next] for each vertex in a face.
    pub fn incident_edges(&self) -> impl Iterator<Item = [T; 3]> + '_
    where
        T: Copy,
    {
        let f = self.as_slice();
        let n = f.len();
        (0..n).map(move |i| [f[(i + n - 1) % n], f[i], f[(i + 1) % n]])
    }
    #[inline]
    pub fn shared_edge(&self, o: &Self) -> Option<[T; 2]>
    where
        T: Eq + Copy,
    {
        self.shared_edges(o).next()
    }
    /// Returns all shared edges between `self` and `o`.
    pub fn shared_edges<'a: 'c, 'b: 'c, 'c>(
        &'a self,
        o: &'b Self,
    ) -> impl Iterator<Item = [T; 2]> + 'c
    where
        T: Eq + Copy,
    {
        self.edges().filter(|&e| {
            o.edges()
                .any(move |[oe0, oe1]| e == [oe0, oe1] || e == [oe1, oe0])
        })
    }
    /// Iterate over triangles in this face rooted at the 0th index.
    pub fn as_triangle_fan(&self) -> impl Iterator<Item = [T; 3]> + '_
    where
        T: Copy + Default,
    {
        let (v0, rest) = if let Some((&v0, rest)) = self.as_slice().split_first() {
            (v0, rest)
        } else {
            (T::default(), [].as_slice())
        };
        rest.array_windows::<2>().map(move |&[v1, v2]| [v0, v1, v2])
    }

    /// Iterate over this face, returning each index with its tri indices
    pub fn iter_with_tri_idxs(
        &self,
    ) -> impl Iterator<Item = (T, Option<Result<[usize; 2], usize>>)> + '_
    where
        T: Copy,
    {
        let (v0, rest) = self
            .as_slice()
            .split_first()
            .map(|(&v, r)| (Some(v), r))
            .unwrap_or_else(|| (None, &[]));
        let (v1, rest) = rest
            .split_first()
            .map(|(&v, r)| (Some(v), r))
            .unwrap_or_else(|| (None, &[]));
        let (vn, rest) = rest
            .split_last()
            .map(|(&v, r)| (Some(v), r))
            .unwrap_or_else(|| (None, &[]));
        let num_tris = self.num_tris();

        let rest_it = rest
            .into_iter()
            .enumerate()
            .map(|(i, &v)| (v, Some(Ok([i, i + 1]))));
        let last_it = vn
            .into_iter()
            .map(move |vn| (vn, Some(Err(num_tris.saturating_sub(1)))));
        v0.into_iter()
            .map(|v0| (v0, None))
            .chain(v1.into_iter().map(|v1| (v1, Some(Err(0)))))
            .chain(rest_it)
            .chain(last_it)
    }

    /// Iterates over all possible triangles rooted at each index.
    pub fn all_triangle_splits(&self) -> impl Iterator<Item = [T; 3]> + '_
    where
        T: Copy,
    {
        let s = self.as_slice();
        let num_vi = s.len();
        (0..num_vi).flat_map(move |v0| {
            ((v0 + 1)..num_vi).filter_map(move |v1| {
                let n = (v1 + 1) % num_vi;
                if v1 <= v0 || n + (num_vi - 3) < v0 || n == v0 {
                    return None;
                }
                assert_ne!(v0, v1);
                assert_ne!(v0, n, "{v0} {v1} {n}");
                Some([v0, v1, n].map(|i| s[i]))
            })
        })
    }
}

impl FaceKind {
    /// Maps the values on this face into either a tri, quad or poly.
    /// CAUTION: may allocate.
    pub fn map_kind<T>(&self, mut f: impl FnMut(usize) -> T) -> FaceKind<T> {
        match self {
            Self::Tri(t) => FaceKind::Tri(t.map(f)),
            Self::Quad(q) => FaceKind::Quad(q.map(f)),
            Self::Poly(p) => FaceKind::Poly(p.iter().map(|&v| f(v)).collect()),
        }
    }
    pub fn from_iter(mut it: impl Iterator<Item = usize>) -> Self {
        let e0 = it.next();
        let e1 = it.next();
        let e2 = it.next();
        let [e0, e1, e2] = match (e0, e1, e2) {
            (None, None, None) => return FaceKind::empty(),
            (Some(e0), None, None) => return FaceKind::Poly(vec![e0]),
            (Some(e0), Some(e1), None) => return FaceKind::Poly(vec![e0, e1]),

            (Some(e0), Some(e1), Some(e2)) => [e0, e1, e2],
            (Some(_), None, Some(_)) | (None, Some(_), Some(_) | None) | (None, None, Some(_)) => {
                unreachable!()
            }
        };
        let q = it.next();
        let penta = it.next();
        let mut poly = match (q, penta) {
            (None, None) => return FaceKind::Tri([e0, e1, e2]),
            (Some(q), None) => return FaceKind::Quad([e0, e1, e2, q]),
            (Some(q), Some(p)) => vec![e0, e1, e2, q, p],
            (None, Some(_)) => unreachable!(),
        };
        poly.extend(it);
        FaceKind::Poly(poly)
    }
    /// For a quad, returns the edge opposite to the provided edge.
    pub fn quad_opp_edge(&self, e0: usize, e1: usize) -> Option<[usize; 2]> {
        match self {
            &Self::Quad([a, b, c, d] | [d, a, b, c] | [c, d, a, b] | [b, c, d, a])
                if (a == e0 && b == e1) || (a == e1 && b == e0) =>
            {
                Some([c, d])
            }
            _ => None,
        }
    }

    #[inline]
    pub fn as_tri(&self) -> Option<[usize; 3]> {
        match self {
            &Self::Tri(tri) => Some(tri),
            _ => None,
        }
    }
    /// Returns each edge in sorted order in this face: minmax(vi0, vi1), minmax(vi1, vi2), ...
    #[inline]
    pub fn edges_ord(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges().map(|[a, b]| std::cmp::minmax(a, b))
    }
    pub fn all_pairs_ord(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        let s = self.as_slice();
        s.iter()
            .enumerate()
            .flat_map(|(i, &v0)| s[(i + 1)..].iter().map(move |&v1| std::cmp::minmax(v0, v1)))
    }

    /// Returns indices of each edge in this face:
    /// [0, 1], [1, 2]... [N, 0]
    pub fn edge_idxs(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        let n = self.len();
        (0..n).map(move |i| [i, (i + 1) % n])
    }
    /// Given a vertex `v` in this face, returns the next vertex.
    /// Panics if `v` is not in this face.
    pub fn next(&self, v: usize) -> usize {
        let f = self.as_slice();
        let pos_v = f.iter().position(|&p| p == v).unwrap();
        f[(pos_v + 1) % f.len()]
    }
    /// Given a vertex `v` in this face, returns the previous vertex.
    /// Panics if `v` is not in this face.
    pub fn prev(&self, v: usize) -> usize {
        let f = self.as_slice();
        let pos_v = f.iter().position(|&p| p == v).unwrap();
        if pos_v == 0 {
            *f.last().unwrap()
        } else {
            f[pos_v - 1]
        }
    }

    /// Remaps each vertex in this face.
    pub fn remap(&mut self, mut f: impl FnMut(usize) -> usize) {
        for v in self.as_mut_slice() {
            *v = f(*v);
        }
    }

    pub fn is_degenerate(&self) -> bool {
        use FaceKind::*;
        match self {
            Tri([a, b, c]) if a == b || b == c || a == c => true,
            Tri(_) => false,
            &Quad([a, b, c, d] | [d, a, b, c] | [c, d, a, b] | [b, c, d, a]) if a == b => {
                Self::Tri([a, c, d]).is_degenerate()
            }
            &Quad([a, _, c, _] | [_, a, _, c]) if a == c => true,
            Quad(_) => false,
            Poly(v) => {
                if v.is_empty() {
                    return true;
                }
                let mut num_uniq = 1;
                let mut end = v.len() - 1;
                while end > 0 && v[end] == v[0] {
                    end -= 1;
                }
                let mut prev_value = usize::MAX;
                for i in 0..end {
                    if v[i] != prev_value {
                        num_uniq += 1;
                    }
                    prev_value = v[i];
                }
                num_uniq > 2
            }
        }
    }

    /// Canonicalize this face, deleting duplicates and retaining order such that the lowest
    /// index vertex is first.
    /// Returns true if this face is now degenerate.
    // ROT indicates whether the output array should be rotated or not
    fn _canonicalize<const ROT: bool>(&mut self) -> bool {
        use FaceKind::*;
        match self {
            Tri([a, b, c]) if a == b || b == c || a == c => return true,
            Tri(_) => {}
            &mut Quad([a, b, c, d] | [d, a, b, c] | [c, d, a, b] | [b, c, d, a]) if a == b => {
                *self = Self::Tri([a, c, d]);
                return self.canonicalize();
            }
            // has a singlet, do not merge
            &mut Quad([a, _, c, _] | [_, a, _, c]) if a == c => return true,

            Quad(_) => {}

            Poly(v) => {
                v.dedup();
                while !v.is_empty() && v.last() == v.first() {
                    v.pop();
                }
                let mut i = 0;
                while v.len() >= 3 && i < v.len() {
                    if v[(i + 2) % v.len()] == v[i] {
                        // remove self and remove next (more robust than removing next element due
                        // to wrapping)
                        v.remove(i);
                        v.remove(i % v.len());
                    } else {
                        i += 1;
                    }
                }
                if v.len() < 3 {
                    return true;
                }
            }
        }
        assert!(self.as_slice().len() > 2);
        if ROT {
            let min_idx = self
                .as_slice()
                .iter()
                .enumerate()
                .min_by_key(|&(_, &v)| v)
                .unwrap()
                .0;
            self.as_mut_slice().rotate_left(min_idx);
        }
        *self = match *self.as_slice() {
            [a, b, c] => Self::Tri([a, b, c]),
            [a, b, c, d] => Self::Quad([a, b, c, d]),
            _ => return false,
        };
        false
    }
    /// Canonicalize this face, deleting duplicates and retaining order such that the lowest
    /// index vertex is first.
    /// Returns true if this face is now degenerate.
    pub fn canonicalize(&mut self) -> bool {
        self._canonicalize::<true>()
    }
    /// Canonicalize this face, deleting duplicates and retaining order such that the lowest
    /// index vertex is first.
    /// Returns true if this face is now degenerate.
    pub fn canonicalize_no_rotate(&mut self) -> bool {
        self._canonicalize::<false>()
    }
    pub(crate) fn insert(&mut self, v: usize) {
        use FaceKind::*;
        *self = match self {
            &mut Tri([a, b, c]) => Quad([a, b, c, v]),
            &mut Quad([a, b, c, d]) => Poly(vec![a, b, c, d, v]),
            Poly(vis) => match vis.len() {
                2 => Tri([vis[0], vis[1], v]),
                3 => Quad([vis[0], vis[1], vis[2], v]),
                _ => {
                    vis.push(v);
                    return;
                }
            },
        }
    }
    pub(crate) fn new(num_verts: usize) -> Self {
        match num_verts {
            3 => Self::Tri([0; 3]),
            4 => Self::Quad([0; 4]),
            _ => Self::Poly(vec![0; num_verts]),
        }
    }

    /// Offset each index in this face by o.
    pub fn offset(&mut self, o: i32) {
        for v in self.as_mut_slice() {
            if o < 0 {
                *v -= o.unsigned_abs() as usize;
            } else {
                *v += o as usize;
            }
        }
    }
    /// The non-normalized normal of this face, using the given set of points
    pub fn normal(&self, v: &[[F; 3]]) -> [F; 3] {
        self.normal_with(|vi| v[vi])
    }

    /// The non-normalized normal of this face, using the provided function for each vertex's
    /// position.
    pub fn normal_with(&self, v: impl Fn(usize) -> [F; 3]) -> [F; 3] {
        match self {
            &FaceKind::Tri([a, b, c]) => cross(sub(v(b), v(a)), sub(v(b), v(c))),
            &FaceKind::Quad([a, b, c, d]) => cross(sub(v(d), v(b)), sub(v(c), v(a))),
            FaceKind::Poly(vs) => {
                let n = vs.len();
                if n == 0 {
                    return [0.; 3];
                }
                let avg = (0..n)
                    .map(|i| {
                        let [p, c, n] = std::array::from_fn(|j| vs[(i + j) % n]);
                        cross(sub(v(n), v(c)), sub(v(p), v(c)))
                    })
                    .fold([0.; 3], add);
                kmul(-1. / (n + 2) as F, avg)
            }
        }
    }

    pub fn area(&self, vs: &[[F; 3]]) -> F {
        match self {
            &FaceKind::Tri(t) => super::tri_area(t.map(|vi| vs[vi])),
            &FaceKind::Quad(q) => super::quad_area(q.map(|vi| vs[vi])),
            FaceKind::Poly(p) => {
                let mut vis = p.iter().copied();
                let Some(root) = vis.next() else {
                    return 0.;
                };
                let Some(mut v0) = vis.next() else {
                    return 0.;
                };
                let mut sum = 0.;
                for v1 in vis {
                    sum += super::tri_area([root, v0, v1].map(|vi| vs[vi]));
                    v0 = v1;
                }
                sum
            }
        }
    }
    /// Compute the centroid of some set of values for this face.
    #[inline]
    pub fn centroid<const N: usize>(&self, vals: &[[F; N]]) -> [F; N] {
        let s = self
            .as_slice()
            .iter()
            .map(|&vi| vals[vi])
            .fold([0.; N], add);
        kmul((self.len() as F).recip(), s)
    }

    /// The AABB for this face given a set of coordinates.
    pub fn aabb<const N: usize>(&self, vals: &[[F; N]]) -> AABB<F, N> {
        let mut out = AABB::default();
        for &vi in self.as_slice() {
            out.add_point(vals[vi]);
        }
        out
    }
}

impl From<&[usize]> for FaceKind {
    fn from(v: &[usize]) -> Self {
        match v {
            &[a, b, c] => Self::Tri([a, b, c]),
            &[a, b, c, d] => Self::Quad([a, b, c, d]),
            o => Self::Poly(o.to_vec()),
        }
    }
}

impl From<Vec<usize>> for FaceKind {
    fn from(v: Vec<usize>) -> Self {
        match *v.as_slice() {
            [a, b, c] => Self::Tri([a, b, c]),
            [a, b, c, d] => Self::Quad([a, b, c, d]),
            _ => Self::Poly(v),
        }
    }
}

/// A barycentric coordinate on a given face, may be a tri, quad or (TODO) polygon,
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Barycentric {
    Tri([F; 3]),
    /// false if first tri, true if second tri. If not contained in either quad, gives the quad
    /// with the largest minimum value
    Quad(bool, [F; 3]),
    /// Index of triangle which contains this barycentric coordinate.
    /// Assumes the input polygon is convex.
    Poly(usize, [F; 3]),
}
impl Barycentric {
    /// Construct a new barycentric coordinate for a specific triangle of a given face.
    pub fn new<T>(f: &FaceKind<T>, ti: usize, b: [F; 3]) -> Self {
        assert!(ti < f.len() - 2);
        match f {
            FaceKind::Tri(_) => Barycentric::Tri(b),
            FaceKind::Quad(_) => Barycentric::Quad(ti == 1, b),
            FaceKind::Poly(_) => Barycentric::Poly(ti, b),
        }
    }
    pub fn tri_idx_and_coords(&self) -> (usize, [F; 3]) {
        match *self {
            Barycentric::Tri(b) => (0, b),
            Barycentric::Quad(f, b) => (f as usize, b),
            Barycentric::Poly(ti, b) => (ti, b),
        }
    }
    pub fn tri(&self, f: &FaceKind) -> [usize; 3] {
        let ti = self.tri_idx_and_coords().0;
        f.as_triangle_fan().nth(ti).unwrap()
    }
    pub fn coords(&self) -> [F; 3] {
        self.tri_idx_and_coords().1
    }
    pub fn coords_mut(&mut self) -> &mut [F; 3] {
        match self {
            Barycentric::Tri(b) => b,
            Barycentric::Quad(_, b) => b,
            Barycentric::Poly(_, b) => b,
        }
    }
    pub fn normalize(&mut self) {
        let sum = self.coords().into_iter().sum::<F>();
        if sum.abs() < 1e-20 {
            return;
        }
        for v in self.coords_mut() {
            *v /= sum;
        }
    }
    /// Returns true if any value is negative for the barycentric.
    pub fn is_outside(&self) -> bool {
        self.coords().iter().any(|&v| v < 0.)
    }

    pub fn tri_idx(&self) -> usize {
        self.tri_idx_and_coords().0
    }

    pub fn clamp(&mut self) {
        // only need to set all to positive, then can normalize correctly.
        for c in self.coords_mut() {
            *c = c.max(0.);
        }
        self.normalize();
    }
}

fn sign(x: F) -> F {
    if x == 0. { 0. } else { x.signum() }
}

#[inline]
fn dot2<const N: usize>(x: [F; N]) -> F {
    dot(x, x)
}

fn tri_sdf_2d(&[p0, p1, p2]: &[[F; 2]; 3], p: [F; 2]) -> F {
    let e0 = sub(p1, p0);
    let e1 = sub(p2, p1);
    let e2 = sub(p0, p2);
    let v0 = sub(p, p0);
    let v1 = sub(p, p1);
    let v2 = sub(p, p2);
    let [pq0, pq1, pq2] = [[v0, e0], [v1, e1], [v2, e2]]
        .map(|[v, e]| sub(v, kmul((dot(v, e) / dot2(e)).clamp(0., 1.), e)));
    let s = sign(cross_2d(e0, e2));
    let dx = dot2(pq0).min(dot2(pq1)).min(dot2(pq2));
    let dy = (s * cross_2d(v0, e0))
        .min(s * cross_2d(v1, e1))
        .min(s * cross_2d(v2, e2));
    -dx.sqrt() * sign(dy)
}

fn tri_sdf_3d(&[a, b, c]: &[[F; 3]; 3], p: [F; 3]) -> F {
    let ba = sub(b, a);
    let pa = sub(p, a);
    let cb = sub(c, b);
    let pb = sub(p, b);
    let ac = sub(a, c);
    let pc = sub(p, c);
    let nor = cross(ba, ac);

    let cond = sign(dot(cross(ba, nor), pa))
        + sign(dot(cross(cb, nor), pb))
        + sign(dot(cross(ac, nor), pc))
        < 2.0;
    let v = if cond {
        let [a, b, c] = [[ba, pa], [cb, pb], [ac, pc]].map(|[e, p]| {
            let v = kmul((dot(e, p) / dot2(e)).clamp(0., 1.), e);
            dot2(sub(v, p))
        });
        a.min(b).min(c)
    } else {
        dot(nor, pa) * dot(nor, pa) / dot2(nor)
    };
    v.sqrt()
}

macro_rules! impl_barycentrics {
    ($barycentric_fn: tt, $dim: tt, $dist_fn: expr) => {
        pub fn barycentric_tri(&self, p: [F; $dim]) -> [F; 3] {
            let &FaceKind::Tri(t) = self else {
                panic!("FaceKind::barycentric_tri requires tri, got {self:?}");
            };
            $barycentric_fn(p, t)
        }
        /// Computes the barycentric coordinate of a point p.
        pub fn barycentric(&self, p: [F; $dim]) -> Barycentric {
            match self {
                &FaceKind::Tri(t) => Barycentric::Tri($barycentric_fn(p, t)),
                &FaceKind::Quad(_) => {
                    // find triangle with minimum signed distance, and compute bary of that
                    // triangle.
                    let (i, t, _) = self
                        .as_triangle_fan()
                        .enumerate()
                        .map(|(i, t)| (i, t, $dist_fn(&t, p)))
                        .min_by(|(_, _, a), (_, _, b)| {
                            a.partial_cmp(&b).unwrap_or_else(|| {
                                panic!("Quad barycentric was not finite {a} {b}")
                            })
                        })
                        .unwrap();
                    Barycentric::Quad(i == 1, $barycentric_fn(p, t))
                }
                FaceKind::Poly(poly) => {
                    assert!(!poly.is_empty());
                    let (i, t, _) = self
                        .as_triangle_fan()
                        .enumerate()
                        .map(|(i, t)| (i, t, $dist_fn(&t, p)))
                        .min_by(|(_, _, a), (_, _, b)| {
                            a.partial_cmp(&b).expect("Poly Barycentric was not finite")
                        })
                        .unwrap();

                    Barycentric::Poly(i, $barycentric_fn(p, t))
                }
            }
        }
        /// If it is known that this is a tri, can be used more efficiently than generic
        /// `from_barycentric`.
        ///
        /// Panics if this is not a tri.
        pub fn from_barycentric_tri(&self, [b0, b1, b2]: [F; 3]) -> [F; $dim] {
            let &FaceKind::Tri([a, b, c]) = self else {
                panic!("FaceKind::from_barycentric_tri requires tri, got {self:?}",);
            };
            add(kmul(b0, a), add(kmul(b1, b), kmul(b2, c)))
        }

        pub fn from_barycentric(&self, b: Barycentric) -> [F; $dim] {
            let (tri_idx, b) = b.tri_idx_and_coords();
            let t = self.as_triangle_fan().nth(tri_idx).unwrap();
            add(kmul(b[0], t[0]), add(kmul(b[1], t[1]), kmul(b[2], t[2])))
        }
    };
}

impl FaceKind<[F; 2]> {
    impl_barycentrics!(barycentric_2d, 2, tri_sdf_2d);
    pub fn scale_by(&mut self, x: F, y: F) {
        for p in self.as_mut_slice() {
            p[0] *= x;
            p[1] *= y;
        }
    }
    pub fn area(&self) -> F {
        match self {
            &FaceKind::Tri(t) => super::tri_area_2d(t),
            &FaceKind::Quad(q) => super::quad_area_2d(q),
            &FaceKind::Poly(_) => todo!(),
        }
    }
    pub fn perimeter(&self) -> F {
        self.edges().map(|[e0, e1]| length(sub(e0, e1))).sum::<F>()
    }
    pub fn tri(&self) -> Option<[[F; 2]; 3]> {
        if let Self::Tri(t) = self {
            return Some(*t);
        };
        None
    }
}

impl FaceKind<[F; 3]> {
    impl_barycentrics!(barycentric_3d, 3, tri_sdf_3d);

    /// The non-normalized normal of this face.
    pub fn normal(&self) -> [F; 3] {
        match self {
            &FaceKind::Tri([a, b, c]) => cross(sub(b, a), sub(b, c)),
            &FaceKind::Quad([a, b, c, d]) => cross(sub(d, b), sub(c, a)),
            FaceKind::Poly(vs) => {
                let n = vs.len();
                if n == 0 {
                    return [0.; 3];
                }
                let avg = (0..n)
                    .map(|i| {
                        let [p, c, n] = std::array::from_fn(|j| vs[(i + j) % n]);
                        cross(sub(n, c), sub(p, c))
                    })
                    .reduce(add)
                    .unwrap();
                kmul(-1. / (n + 2) as F, avg)
            }
        }
    }
    /// The area of this face
    pub fn area(&self) -> F {
        match self {
            &FaceKind::Tri(t) => super::tri_area(t),
            &FaceKind::Quad(q) => super::quad_area(q),
            FaceKind::Poly(p) => {
                let mut vis = p.iter().copied();
                let Some(root) = vis.next() else {
                    return 0.;
                };
                let Some(mut v0) = vis.next() else {
                    return 0.;
                };
                let mut sum = 0.;
                for v1 in vis {
                    sum += super::tri_area([root, v0, v1]);
                    v0 = v1;
                }
                sum
            }
        }
    }
    pub fn tri(&self) -> Option<[[F; 3]; 3]> {
        if let Self::Tri(t) = self {
            return Some(*t);
        };
        None
    }
}

#[test]
fn test_normal_orientation() {
    // T = [F; 3]
    use super::dot;
    let t = FaceKind::Tri([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]);
    let q = FaceKind::Quad([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]);
    let d = dot(t.normal(), q.normal());
    assert!(d > 0., "{d}");
    let p = FaceKind::Poly(vec![
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [-0.1, 0.5, 0.],
    ]);
    let d = dot(t.normal(), p.normal());
    assert!(d > 0., "{d}");

    // T = [F; 2]
    let t = FaceKind::Tri([[0., 0.], [1., 0.], [1., 1.]]);
    let q = FaceKind::Quad([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
    assert_eq!(t.area().signum(), q.area().signum());
    // No check for polygons yet

    // Indexed
    let vs = [
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [-0.1, 0.5, 0.],
    ];
    let t = FaceKind::Tri([0, 1, 2]);
    let q = FaceKind::Quad([0, 1, 2, 3]);
    let d = dot(t.normal(&vs), q.normal(&vs));
    assert!(d > 0., "{d}");
    let p = FaceKind::Poly(vec![0, 1, 2, 3, 4]);
    let d = dot(t.normal(&vs), p.normal(&vs));
    assert!(d > 0., "{d}");

    // Closure fn
    let d = dot(t.normal_with(|vi| vs[vi]), q.normal_with(|vi| vs[vi]));
    assert!(d > 0., "{d}");
    let p = FaceKind::Poly(vec![0, 1, 2, 3, 4]);
    let d = dot(t.normal_with(|vi| vs[vi]), p.normal_with(|vi| vs[vi]));
    assert!(d > 0., "{d}");
}

#[test]
fn test_all_tri_splits_quad() {
    let q = FaceKind::Quad([0, 1, 2, 3]);
    assert_eq!(
        q.all_triangle_splits().count(),
        4,
        "{:?}",
        q.all_triangle_splits().collect::<Vec<_>>()
    );

    let p = FaceKind::Poly(vec![0, 1, 2, 3, 4]);
    assert_eq!(
        p.all_triangle_splits().count(),
        8,
        "{:?}",
        p.all_triangle_splits().collect::<Vec<_>>()
    );
}

#[test]
fn test_canonicalize_wrap() {
    let mut q = FaceKind::Poly(vec![0, 1, 2, 3, 1]);
    q.canonicalize();
    assert_eq!(q, FaceKind::Tri([1, 2, 3]));
    let mut q = FaceKind::Poly(vec![0, 3, 7, 3, 1, 2, 3]);
    q.canonicalize();
    assert_eq!(q, FaceKind::Tri([1, 2, 3]));
}
