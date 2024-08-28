#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]

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

/// Fuse vertices of a mesh together by distance.
pub mod fuse;
