#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]
#![feature(iter_array_chunks)]
#![feature(cfg_match)]

#[cfg(all(not(feature = "f64"), not(feature = "num-rational")))]
pub type F = f32;
#[cfg(all(not(feature = "f64"), not(feature = "num-rational")))]
fn into(f: F) -> f64 {
    f as f64
}

#[cfg(all(feature = "f64", not(feature = "num-rational")))]
pub type F = f64;
#[cfg(all(feature = "f64", not(feature = "num-rational")))]
fn into(f: F) -> f64 {
    f
}

#[cfg(all(feature = "f64", feature = "num-rational"))]
pub type F = num_rational::Rational64;
#[cfg(all(feature = "f64", feature = "num-rational"))]
fn into(f: F) -> f64 {
    let (n, d) = f.into_raw();
    n as f64 / d as f64
}

#[cfg(all(not(feature = "f64"), feature = "num-rational"))]
pub type F = num_rational::Rational32;
#[cfg(all(not(feature = "f64"), feature = "num-rational"))]
fn into(f: F) -> f64 {
    let (n, d) = f.into_raw();
    n as f64 / d as f64
}

#[cfg(all(feature = "num-rational", feature = "gltf"))]
compile_error!("Rational and GLTF features are mutually exclusive for pars3d.");

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

/// Fuse vertices of a mesh together by distance.
#[cfg(not(feature = "num-rational"))]
pub mod fuse;

#[cfg(feature = "gltf")]
pub mod gltf;

pub mod ply;

/// Unified mesh representation.
pub mod mesh;

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
}
