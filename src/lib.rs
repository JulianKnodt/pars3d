#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]

pub type F = f32;
pub type Vector<const N: usize, T = F> = [T; N];
pub type Vec3 = Vector<3>;
pub type Vec2 = Vector<2>;

/// OBJ parsing
pub mod obj;

/// OFF parsing
pub mod off;

/// STL parsing
pub mod stl;
