#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]
#![feature(array_try_map)]
#![feature(iter_array_chunks)]
#![feature(cfg_match)]
#![feature(cmp_minmax)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(let_chains)]
#![feature(array_chunks)]
#![feature(ascii_char)]
#![feature(ascii_char_variants)]
#![feature(assert_matches)]
#![feature(generic_arg_infer)]
#![feature(os_str_display)]

#[cfg(not(feature = "f64"))]
pub type U = u32;
#[cfg(not(feature = "f64"))]
pub type I = i32;

#[cfg(feature = "f64")]
pub type U = u64;
#[cfg(feature = "f64")]
pub type I = i64;

use std::path::Path;

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

/// MERL dataset loading.
pub mod merl;

/// Fuse vertices of a mesh together by distance.
pub mod fuse;

/// Coloring data for visualizing features.
pub mod coloring;

/// Visualize per-element attributes.
pub mod visualization;

/// Load GLTF meshes.
#[cfg(feature = "gltf")]
pub mod gltf;

/// Animation related structs
pub mod anim;

/// Unified mesh representation.
pub mod mesh;

/// Approximately convert a triangle mesh to a mixed tri/quad mesh.
pub mod tri_to_quad;

#[cfg(feature = "fbx")]
/// FBX parsing.
pub mod fbx;

pub mod util;

/// Quasi-random generation
mod rand;

pub fn load(v: impl AsRef<Path>) -> std::io::Result<mesh::Scene> {
    use util::FileFormat::*;
    let scene = match util::extension_to_format(&v) {
        OBJ => obj::parse(v, false, false)?.into(),
        FBX => fbx::parser::load(v)?.into(),

        #[cfg(feature = "gltf")]
        GLB => gltf::load(v).map_err(std::io::Error::other)?.into(),
        #[cfg(not(feature = "gltf"))]
        GLB => return Err(std::io::Error::other("Not compiled with GLTF support")),

        PLY => mesh::Mesh::from(ply::Ply::read_from_file(v)?).into_scene(),
        STL => mesh::Mesh::from(stl::read_from_file(v)?).into_scene(),
        OFF => mesh::Mesh::from(off::read_from_file(v)?).into_scene(),
        Unknown => return Err(std::io::Error::other("Don't know how to load")),
    };
    Ok(scene)
}

pub fn save(v: impl AsRef<Path>, scene: &mesh::Scene) -> std::io::Result<()> {
    use util::FileFormat::*;
    match util::extension_to_format(&v) {
        OBJ => obj::save_obj(
            scene,
            v,
            |mtl_file| obj::OutputKind::New(mtl_file.into()),
            |_tk, s| obj::OutputKind::New(s.into()),
        ),
        FBX => {
            let scene: mesh::Scene = scene.clone();
            let fbx_scene: fbx::FBXScene = scene.into();
            let f = std::fs::File::create(v)?;
            let buf = std::io::BufWriter::new(f);
            fbx::export::export_fbx(&fbx_scene, buf)
        }
        #[cfg(feature = "gltf")]
        GLB => {
            let f = std::fs::File::create(v)?;
            let buf = std::io::BufWriter::new(f);
            gltf::save_glb(scene, buf)
        }
        #[cfg(not(feature = "gltf"))]
        GLB => Err(std::io::Error::other("Not compiled with GLTF support")),

        PLY => {
            let f = std::fs::File::create(v)?;
            let buf = std::io::BufWriter::new(f);
            let p: ply::Ply = scene.into_flattened_mesh().into();
            p.write(buf)
        }
        STL => {
            let f = std::fs::File::create(v)?;
            let buf = std::io::BufWriter::new(f);
            let s: stl::STL = scene.into_flattened_mesh().into();
            stl::write(&s, buf)
        }
        OFF => {
            let f = std::fs::File::create(v)?;
            let buf = std::io::BufWriter::new(f);
            let o: off::OFF = scene.into_flattened_mesh().into();
            o.write(buf)
        }
        Unknown => Err(std::io::Error::other("Don't know how to save")),
    }
}

/// Face representation for meshes.
/// Tris and quads are stack allocated,
/// If you're a madman and store general polygons they're on the heap.
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
        self.as_slice().len().saturating_sub(2)
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
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
    /// `true` if there are any vertices in this face.
    pub fn is_empty(&self) -> bool {
        use FaceKind::*;
        match self {
            Tri(_) => false,
            Quad(_) => false,
            Poly(v) => v.is_empty(),
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

pub fn length<const N: usize>(v: [F; N]) -> F {
    v.iter().map(|v| v * v).sum::<F>().max(0.)
}

/// Normalizes a vector, returning a zero vector if it has 0 norm
#[inline]
pub fn normalize<const N: usize>(v: [F; N]) -> [F; N] {
    let sum: F = v.iter().map(|v| v * v).sum();
    if sum < 1e-20 {
        return [0.; N];
    }
    let s = sum.sqrt();
    v.map(|v| v / s)
}

pub(crate) fn dot([a, b, c]: [F; 3], [x, y, z]: [F; 3]) -> F {
    a * x + b * y + c * z
}

pub(crate) fn append_one([a, b, c]: [F; 3]) -> [F; 4] {
    [a, b, c, 1.]
}

pub(crate) fn quad_area([a, b, c, d]: [[F; 3]; 4]) -> F {
    0.5 * length(cross(sub(a, c), sub(b, d)))
}

pub fn tri_area([a, b, c]: [[F; 3]; 3]) -> F {
    0.5 * length(cross(sub(a, b), sub(b, c)))
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

/// Identity Matrix
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

pub fn rotate_on_axis(v: [F; 3], axis: [F; 3], s: F, c: F) -> [F; 3] {
    let r = add(kmul(c, v), kmul(dot(v, axis) * (1. - c), axis));
    add(r, kmul(s, cross(axis, v)))
}

pub(crate) fn edges(vis: &[usize]) -> impl Iterator<Item = [usize; 2]> + '_ {
    (0..vis.len()).map(|vi| [vis[vi], vis[(vi + 1) % vis.len()]])
}
