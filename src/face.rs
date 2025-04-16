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

    /// Returns an empty face
    pub(crate) fn empty() -> Self {
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
}

impl FaceKind {
    pub fn map_kind<T>(&self, f: impl Fn(usize) -> T) -> FaceKind<T> {
        match self {
            Self::Tri(t) => FaceKind::Tri(t.map(f)),
            Self::Quad(q) => FaceKind::Quad(q.map(f)),
            Self::Poly(p) => FaceKind::Poly(p.into_iter().map(|&v| f(v)).collect()),
        }
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

    /// Returns each edge in this face: [vi0, vi1], [vi1, vi2]... [viN, vi0]
    pub fn edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        edges(self.as_slice())
    }
    /// Returns each edge in sorted order in this face: minmax(vi0, vi1), minmax(vi1, vi2), ...
    pub fn edges_ord(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.edges().map(|[a, b]| std::cmp::minmax(a, b))
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
    /// Iterate over triangles in this face rooted at the 0th index.
    pub fn as_triangle_fan(&self) -> impl Iterator<Item = [usize; 3]> + '_ {
        let (&v0, rest) = self.as_slice().split_first().unwrap();
        rest.array_windows::<2>().map(move |&[v1, v2]| [v0, v1, v2])
    }

    /// Remaps each vertex in this face.
    pub fn map(&mut self, mut f: impl FnMut(usize) -> usize) {
        for v in self.as_mut_slice() {
            *v = f(*v);
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
            Quad(_) => {}
            Poly(ref mut v) => {
                v.dedup();
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

impl FaceKind<[F; 2]> {
    /// Computes the barycentric coordinate of a 2D point p.
    pub fn barycentric(&self, p: [F; 2]) -> [F; 3] {
        match self {
            FaceKind::Tri(t) => barycentric_2d(p, *t),
            FaceKind::Quad(_) => todo!(),
            FaceKind::Poly(_p) => unimplemented!(),
        }
    }
    pub fn from_barycentric(&self, bs: [F; 3]) -> [F; 2] {
        match self {
            FaceKind::Tri(ps) => ps
                .iter()
                .enumerate()
                .fold([0., 0.], |acc, (i, n)| add(acc, kmul(bs[i], *n))),
            FaceKind::Quad(_) => todo!(),
            FaceKind::Poly(_) => unimplemented!(),
        }
    }
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
}

impl FaceKind<[F; 3]> {
    pub fn barycentric(&self, p: [F; 3]) -> [F; 3] {
        match self {
            FaceKind::Tri(t) => barycentric_3d(p, *t),
            FaceKind::Quad(_) => todo!(),
            FaceKind::Poly(_p) => unimplemented!(),
        }
    }
    pub fn from_barycentric(&self, bs: [F; 3]) -> [F; 3] {
        match self {
            FaceKind::Tri(ps) => ps
                .iter()
                .enumerate()
                .fold([0.; 3], |acc, (i, n)| add(acc, kmul(bs[i], *n))),
            FaceKind::Quad(_) => todo!(),
            FaceKind::Poly(_) => unimplemented!(),
        }
    }
    /// The non-normalized normal of this face.
    pub fn normal(&self) -> [F; 3] {
        match self {
            &FaceKind::Tri([a, b, c]) => cross(sub(b, a), sub(b, c)),
            &FaceKind::Quad([a, b, c, d]) => cross(sub(c, a), sub(d, b)),
            FaceKind::Poly(vs) => {
                let n = vs.len();
                let avg = (0..n)
                    .map(|i| {
                        let [p, c, n] = std::array::from_fn(|j| vs[(i + j) % n]);
                        cross(sub(n, c), sub(p, c))
                    })
                    .reduce(add)
                    .unwrap();
                kmul(1. / (n + 2) as F, avg)
            }
        }
    }
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
