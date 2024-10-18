#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]
#![feature(iter_array_chunks)]
#![feature(cfg_match)]
#![feature(cmp_minmax)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(let_chains)]
#![feature(array_chunks)]

#[cfg(not(feature = "f64"))]
pub type F = f32;

#[cfg(feature = "f64")]
pub type F = f64;

/// Alias for array of floats.
pub type Vector<const N: usize, T = F> = [T; N];

pub type Vec3 = Vector<3>;
pub type Vec2 = Vector<2>;

/// OBJ parsing
pub mod obj;

/// OFF parsing
pub mod off;

/// STL parsing
pub mod stl;

/// PLY parsing
pub mod ply;

/// Fuse vertices of a mesh together by distance.
pub mod fuse;

/// Coloring data for visualizing features.
pub mod coloring;

/// Visualize per-element attributes.
pub mod visualization;

/// Load GLTF meshes.
#[cfg(feature = "gltf")]
pub mod gltf;

/// Unified mesh representation.
pub mod mesh;

/// Approximately convert a triangle mesh to a mixed tri/quad mesh.
pub mod tri_to_quad;

/// FBX parsing.
#[cfg(feature = "fbx")]
pub mod fbx;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceKind {
    Tri([usize; 3]),
    Quad([usize; 4]),
    Poly(Vec<usize>),
}

impl FaceKind {
    pub fn as_slice(&self) -> &[usize] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_slice(),
            Quad(q) => q.as_slice(),
            Poly(v) => v.as_slice(),
        }
    }
    pub fn num_tris(&self) -> usize {
        self.as_slice().len() - 2
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        use FaceKind::*;
        match self {
            Tri(t) => t.as_mut_slice(),
            Quad(q) => q.as_mut_slice(),
            Poly(v) => v.as_mut_slice(),
        }
    }
    pub fn len(&self) -> usize {
        use FaceKind::*;
        match self {
            Tri(_) => 3,
            Quad(_) => 4,
            Poly(v) => v.len(),
        }
    }
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
    pub fn is_empty(&self) -> bool {
        use FaceKind::*;
        match self {
            Tri(_) => false,
            Quad(_) => false,
            Poly(v) => v.is_empty(),
        }
    }
    /// Iterate over triangles in this face rooted at the 0th index.
    pub fn as_triangle_fan(&self) -> impl Iterator<Item = [usize; 3]> + '_ {
        let (&v0, rest) = self.as_slice().split_first().unwrap();
        rest.array_windows::<2>().map(move |&[v1, v2]| [v0, v1, v2])
    }
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
}

pub(crate) fn kmul<const N: usize>(k: F, v: [F; N]) -> [F; N] {
    v.map(|v| v * k)
}

pub(crate) fn add<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] + b[i])
}

pub(crate) fn sub([a, b, c]: [F; 3], [x, y, z]: [F; 3]) -> [F; 3] {
    [a - x, b - y, c - z]
}

pub(crate) fn cross([x, y, z]: [F; 3], [a, b, c]: [F; 3]) -> [F; 3] {
    [y * c - z * b, z * a - x * c, x * b - y * a]
}

pub(crate) fn dot([a, b, c]: [F; 3], [x, y, z]: [F; 3]) -> F {
    a * x + b * y + c * z
}

pub(crate) fn append_one([a, b, c]: [F; 3]) -> [F; 4] {
    [a, b, c, 1.]
}

/// Apply a transformation (col major 4x4) to a point
pub fn tform_point(tform: [[F; 4]; 4], p: [F; 3]) -> [F; 3] {
    let out = (0..4)
        .map(|i| {
            if i == 3 {
                tform[i]
            } else {
                kmul(p[i], tform[i])
            }
        })
        .fold([0.; 4], add);
    assert_ne!(out[3], 0., "{tform:?}*{p:?} = {out:?}");
    std::array::from_fn(|i| out[i] / out[3])
}
pub fn identity<const N: usize>() -> [[F; N]; N] {
    let mut out = [[0.; N]; N];
    for i in 0..N {
        out[i][i] = 1.;
    }
    out
}

/// Matrix multiplication.
/// For composing transforms together.
pub fn matmul<const N: usize>(ta: [[F; N]; N], tb: [[F; N]; N]) -> [[F; N]; N] {
    let mut out = [[0.; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                out[i][j] += ta[i][k] * tb[k][j];
            }
        }
    }
    out
}

#[inline]
pub fn normalize<const N: usize>(v: [F; N]) -> [F; N] {
    let sum: F = v.iter().map(|v| v * v).sum();
    if sum < 1e-20 {
        return [0.; N];
    }
    let s = sum.sqrt();
    v.map(|v| v / s)
}

pub(crate) fn edges(vis: &[usize]) -> impl Iterator<Item = [usize; 2]> + '_ {
    (0..vis.len()).map(|vi| [vis[vi], vis[(vi + 1) % vis.len()]])
}
