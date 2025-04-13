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

/// Edge representations
pub mod edge;

pub use mesh::Mesh;
pub use mesh::Scene;

/// Approximately convert a triangle mesh to a mixed tri/quad mesh.
pub mod tri_to_quad;

#[cfg(feature = "fbx")]
/// FBX parsing.
pub mod fbx;

pub mod util;

/// Re-exported for materials.
pub use image;

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

pub mod face;
pub use face::FaceKind;

pub(crate) fn kmul<const N: usize>(k: F, v: [F; N]) -> [F; N] {
    v.map(|v| v * k)
}

pub(crate) fn add<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] + b[i])
}

pub(crate) fn sub<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] - b[i])
}

pub(crate) fn cross([x, y, z]: [F; 3], [a, b, c]: [F; 3]) -> [F; 3] {
    [y * c - z * b, z * a - x * c, x * b - y * a]
}

pub fn length<const N: usize>(v: [F; N]) -> F {
    v.iter().map(|v| v * v).sum::<F>().max(0.)
}

/// Normalizes a vector, returning a zero vector if it has 0 norm
#[inline]
pub(crate) fn normalize<const N: usize>(v: [F; N]) -> [F; N] {
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

pub fn quad_area([a, b, c, d]: [[F; 3]; 4]) -> F {
    0.5 * length(cross(sub(a, c), sub(b, d)))
}

pub fn tri_area([a, b, c]: [[F; 3]; 3]) -> F {
    0.5 * length(cross(sub(a, b), sub(b, c)))
}

/// More robust ln triangle area.
pub fn ln_tri_area([a,b,c]: [[F;3]; 3]) -> F {
  let e0 = length(sub(a,b));
  let e1 = length(sub(b,c));
  let e2 = length(sub(c,a));
  let s = 0.5 * (e0 + e1 + e2);
  let acl = |s: F| (s.abs() + 1e-14).ln();
  0.5 * (acl(s) + acl(s-e0) + acl(s-e1) + acl(s-e2))
}

pub fn tri_area_2d([a, b, c]: [[F; 2]; 3]) -> F {
    let [ba0, ba1] = sub(b, a);
    let [ca0, ca1] = sub(c, a);
    0.5 * (ba0 * ca1 - ba1 * ca0)
}

/// Given a triangle in 2d and a point p, compute the barycentric coordinate of point `p`.
pub fn barycentric_2d(p: [F; 2], [a, b, c]: [[F; 2]; 3]) -> [F; 3] {
    let a2 = tri_area_2d([a, b, p]);
    let a1 = tri_area_2d([a, p, c]);
    let a0 = tri_area_2d([p, b, c]);

    let total_area = a0 + a1 + a2;
    if total_area.abs() < 1e-12 {
        return [1., 0., 0.];
    }

    [a0, a1, a2].map(|v| v / total_area)
}

#[test]
fn test_bary_2d() {
    let tri = [[0., 0.], [1., 0.], [0., 1.]];

    assert_eq!(barycentric_2d([0., 0.], tri), [1., 0., 0.]);
    assert_eq!(barycentric_2d([1., 0.], tri), [0., 1., 0.]);
    assert_eq!(barycentric_2d([0., 1.], tri), [0., 0., 1.]);
}

pub fn barycentric_3d(p: [F; 3], [a, b, c]: [[F; 3]; 3]) -> [F; 3] {
    let a2 = tri_area([a, b, p]);
    let a1 = tri_area([a, p, c]);
    let a0 = tri_area([p, b, c]);

    let total_area = a0 + a1 + a2;
    if total_area.abs() < 1e-12 {
        return [1., 0., 0.];
    }

    [a0, a1, a2].map(|v| v / total_area)
}

#[test]
fn test_bary_3d() {
    let tri = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];

    assert_eq!(barycentric_3d([0., 0., 0.], tri), [1., 0., 0.]);
    assert_eq!(barycentric_3d([1., 0., 0.], tri), [0., 1., 0.]);
    assert_eq!(barycentric_3d([0., 1., 0.], tri), [0., 0., 1.]);
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

pub fn edges(vis: &[usize]) -> impl Iterator<Item = [usize; 2]> + '_ {
    (0..vis.len()).map(|vi| [vis[vi], vis[(vi + 1) % vis.len()]])
}
