#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]
#![feature(iter_array_chunks)]
#![feature(cfg_match)]
#![feature(cmp_minmax)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(let_chains)]

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

#[cfg(feature = "gltf")]
pub mod gltf;

/// Unified mesh representation.
pub mod mesh;

/// Approximately convert a triangle mesh to a mixed tri/quad mesh.
pub mod tri_to_quad;

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
    pub fn is_empty(&self) -> bool {
        use FaceKind::*;
        match self {
            Tri(_) => false,
            Quad(_) => false,
            Poly(v) => v.is_empty(),
        }
    }
}

pub(crate) fn kmul(k: F, [x, y, z]: [F; 3]) -> [F; 3] {
    [x * k, y * k, z * k]
}

pub(crate) fn add([a, b, c]: [F; 3], [x, y, z]: [F; 3]) -> [F; 3] {
    [a + x, b + y, c + z]
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
