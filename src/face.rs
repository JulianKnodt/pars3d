use crate::{add, barycentric_2d, barycentric_3d, cross, edges, kmul, sub, F};

/// Face representation for meshes.
/// Tris and quads are stack allocated,
/// If you're a madman and store general polygons they're on the heap.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceKind<T = usize> {
    Tri([T; 3]),
    Quad([T; 4]),
    Poly(Vec<T>),
}

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
    #[inline]
    pub fn shared_edge(&self, o: &Self) -> Option<[T; 2]>
    where
        T: Eq + Copy,
    {
        for e in self.edges() {
            for [oe0, oe1] in o.edges() {
                if e == [oe0, oe1] || e == [oe1, oe0] {
                    return Some(e);
                }
            }
        }
        return None;
    }
    /// Iterate over triangles in this face rooted at the 0th index.
    pub fn as_triangle_fan(&self) -> impl Iterator<Item = [T; 3]> + '_
    where
        T: Copy,
    {
        let (&v0, rest) = self.as_slice().split_first().unwrap();
        rest.array_windows::<2>().map(move |&[v1, v2]| [v0, v1, v2])
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
            Self::Poly(p) => FaceKind::Poly(p.into_iter().map(|&v| f(v)).collect()),
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
            Tri([a, b, c]) if a == b || b == c || a == c => return true,
            Tri(_) => false,
            &Quad([a, b, c, d] | [d, a, b, c] | [c, d, a, b] | [b, c, d, a]) if a == b => {
                return Self::Tri([a, c, d]).is_degenerate()
            }
            &Quad([a, _, c, _] | [_, a, _, c]) if a == c => return true,
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
    pub fn canonicalize(&mut self) -> bool {
        use FaceKind::*;
        match self {
            Tri([a, b, c]) if a == b || b == c || a == c => return true,
            Tri(_) => {}
            &mut Quad([a, b, c, d] | [d, a, b, c] | [c, d, a, b] | [b, c, d, a]) if a == b => {
                *self = Self::Tri([a, c, d]);
                return self.canonicalize();
            }
            &mut Quad([a, _, c, _] | [_, a, _, c]) if a == c => return true,
            Quad(_) => {}
            Poly(ref mut v) => {
                v.dedup();
                // TODO here also needs to handle the case where repeats are separated by 1.
                while !v.is_empty() && v.last() == v.first() {
                    v.pop();
                }
                if v.len() < 3 {
                    return true;
                }
            }
        }
        assert!(self.as_slice().len() > 2);
        let min_idx = self
            .as_slice()
            .iter()
            .enumerate()
            .min_by_key(|&(_, &v)| v)
            .unwrap()
            .0;
        self.as_mut_slice().rotate_left(min_idx);
        *self = match self.as_slice() {
            &[a, b, c] => Self::Tri([a, b, c]),
            &[a, b, c, d] => Self::Quad([a, b, c, d]),
            _ => return false,
        };
        false
    }
    pub(crate) fn insert(&mut self, v: usize) {
        use FaceKind::*;
        *self = match self {
            &mut Tri([a, b, c]) => Quad([a, b, c, v]),
            &mut Quad([a, b, c, d]) => Poly(vec![a, b, c, d, v]),
            Poly(ref mut vis) => match vis.len() {
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
    pub fn offset(&mut self, o: i32) {
        for v in self.as_mut_slice() {
            if o < 0 {
                *v = *v - (o.abs() as usize);
            } else {
                *v = *v + o as usize;
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
    pub fn tri_idx_and_coords(&self) -> (usize, [F; 3]) {
        match *self {
            Barycentric::Tri(b) => (0, b),
            Barycentric::Quad(f, b) => (f as usize, b),
            Barycentric::Poly(ti, b) => (ti, b),
        }
    }
    pub fn coords(&self) -> [F; 3] {
        self.tri_idx_and_coords().1
    }
    /// Returns true if any value is negative for the barycentric.
    pub fn is_outside(&self) -> bool {
        self.coords().iter().any(|&v| v < 0.)
    }
}

macro_rules! impl_barycentrics {
    ($barycentric_fn: tt, $dim: tt) => {
        pub fn barycentric_tri(&self, p: [F; $dim]) -> [F; 3] {
            let &FaceKind::Tri(t) = self else {
                panic!("FaceKind::barycentric_tri requires tri, got {self:?}");
            };
            $barycentric_fn(p, t)
        }
        /// Computes the barycentric coordinate of a 2D point p.
        pub fn barycentric(&self, p: [F; $dim]) -> Barycentric {
            match self {
                &FaceKind::Tri(t) => Barycentric::Tri($barycentric_fn(p, t)),
                &FaceKind::Quad(_) => {
                    let (i, b, _) = self
                        .as_triangle_fan()
                        .enumerate()
                        .map(|(i, t)| {
                            let b = $barycentric_fn(p, t);
                            (i, b, b[0].min(b[1]).min(b[2]))
                        })
                        .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(&b).unwrap())
                        .unwrap();
                    Barycentric::Quad(i == 1, b)
                }
                FaceKind::Poly(poly) => {
                    assert!(!poly.is_empty());
                    let (i, b, _) = self
                        .as_triangle_fan()
                        .enumerate()
                        .map(|(i, t)| {
                            let b = $barycentric_fn(p, t);
                            (i, b, b[0].min(b[1]).min(b[2]))
                        })
                        .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(&b).unwrap())
                        .unwrap();

                    Barycentric::Poly(i, b)
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
            t.iter()
                .enumerate()
                .map(|(i, n)| kmul(b[i], *n))
                .fold([0.; _], add)
        }
    };
}

impl FaceKind<[F; 2]> {
    impl_barycentrics!(barycentric_2d, 2);
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
    pub fn tri(&self) -> Option<[[F; 2]; 3]> {
        if let Self::Tri(t) = self {
            return Some(*t);
        };
        None
    }
}

impl FaceKind<[F; 3]> {
    impl_barycentrics!(barycentric_3d, 3);

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
