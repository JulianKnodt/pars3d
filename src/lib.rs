#![feature(str_split_whitespace_remainder)]
#![feature(array_windows)]
#![feature(array_try_map)]
#![feature(iter_array_chunks)]
#![feature(cmp_minmax)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(array_chunks)]
#![feature(ascii_char)]
#![feature(ascii_char_variants)]
#![feature(assert_matches)]

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

#[cfg(feature = "fbx")]
/// FBX parsing.
pub mod fbx;

/// Load GLTF meshes.
#[cfg(feature = "gltf")]
pub mod gltf;

/// Use with caution (.wrl files)
pub mod vrml;

// BRDF ---

/// MERL dataset loading.
pub mod merl;

// UTILITIES ---

/// Fuse vertices of a mesh together by distance.
pub mod fuse;

/// Coloring data for visualizing features.
pub mod coloring;

/// Utilities for generating noise.
pub mod noise;

/// Functions for interpolation over [0,1]
pub mod func;

/// Visualize per-element attributes.
pub mod visualization;

/// Trace lines on the surface of a mesh.
/// For visualization or for art.
pub mod tracing;

// GEOMETRY PROCESSING ---

/// Animation related structs
pub mod anim;

/// Unified mesh representation
pub mod mesh;

/// Geometry processing on meshes. Often allocates.
pub mod geom_processing;

/// Edge representations
pub mod edge;

/// Vertex Adjacency utilities
pub mod adjacency;

pub use mesh::Mesh;
pub use mesh::Scene;

/// Approximately convert a triangle mesh to a mixed tri/quad mesh.
pub mod tri_to_quad;

pub mod util;

/// Quaternion related stuff
pub mod quat;

/// AABB (mostly for UVs)
pub mod aabb;

/// Face Representation
pub mod face;
pub use face::FaceKind;

/// SVG image saving for UV texture maps
#[cfg(feature = "svg")]
pub mod svg;

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

pub fn kmul<const N: usize>(k: F, v: [F; N]) -> [F; N] {
    v.map(|v| v * k)
}

pub(crate) fn add<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] + b[i])
}

pub(crate) fn sub<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    std::array::from_fn(|i| a[i] - b[i])
}

pub fn dist<const N: usize>(a: [F; N], b: [F; N]) -> F {
    length(sub(b, a))
}

pub(crate) fn cross([x, y, z]: [F; 3], [a, b, c]: [F; 3]) -> [F; 3] {
    [
        diff_of_prod(y, c, z, b),
        diff_of_prod(z, a, x, c),
        diff_of_prod(x, b, y, a),
    ]
    //[y * c - z * b, z * a - x * c, x * b - y * a]
}

pub(crate) fn cross_2d([x, y]: [F; 2], [a, b]: [F; 2]) -> F {
    diff_of_prod(x, b, y, a)
}

pub(crate) fn tri_area_2d([a, b, c]: [[F; 2]; 3]) -> F {
    cross_2d(sub(b, a), sub(c, a)) / 2.
}

pub(crate) fn quad_area_2d([a, b, c, d]: [[F; 2]; 4]) -> F {
    cross_2d(sub(a, c), sub(b, d)) / 2.
}

pub fn length<const N: usize>(v: [F; N]) -> F {
    v.iter().map(|v| v * v).sum::<F>().max(0.).sqrt()
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

pub(crate) fn dot<const N: usize>(a: [F; N], b: [F; N]) -> F {
    let mut out = 0.;
    for i in 0..N {
        out += a[i] * b[i];
    }
    out
}

pub(crate) fn append_one([a, b, c]: [F; 3]) -> [F; 4] {
    [a, b, c, 1.]
}

pub fn quad_area([a, b, c, d]: [[F; 3]; 4]) -> F {
    0.5 * length(cross(sub(b, d), sub(a, c)))
}

pub fn tri_area([a, b, c]: [[F; 3]; 3]) -> F {
    0.5 * length(cross(sub(a, b), sub(b, c)))
}
pub(crate) fn tri_normal([a, b, c]: [[F; 3]; 3]) -> [F; 3] {
    cross(sub(a, b), sub(b, c))
}

/// Heron's formula for computing the area of triangles in N-dimensional space.
pub fn tri_area_nd<const N: usize>([a, b, c]: [[F; N]; 3]) -> F {
    let e0 = length(sub(b, a));
    let e1 = length(sub(c, b));
    let e2 = length(sub(a, c));
    let s = (e0 + e1 + e2) / 2.;
    (s * (s - e0) * (s - e1) * (s - e2)).sqrt()
}

/// Given 3 triangle edge lengths, compute the cosine for the angle opposite to each edge.
pub fn cosine_angles([a, b, c]: [F; 3]) -> [F; 3] {
    if a == 0. || b == 0. || c == 0. {
        // angles are 0, cosine of 0 = 1
        return [1.; 3];
    }

    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let a_ang = -(a2 - b2 - c2) / (2. * b * c);
    let b_ang = -(b2 - a2 - c2) / (2. * a * c);
    let c_ang = -(c2 - b2 - a2) / (2. * b * a);
    [a_ang, b_ang, c_ang]
}

/// Given three edge lengths, compute the area of the triangle with those lengths.
/// More numerically stable than a naive implementation, relies on computing the product of the
/// semi-perimeter in log-space
pub fn herons_area([e0, e1, e2]: [F; 3]) -> F {
    if e0 == 0. || e1 == 0. || e2 == 0. {
        return 0.;
    }
    let s = (e0 + e1 + e2) / 2.;
    let f = |v: F| v.max(1e-14).ln();
    let v = f(s) + f(s - e0) + f(s - e1) + f(s - e2);
    assert!(v.is_finite(), "{} {} {}", s - e0, s - e1, s - e2);
    (v * 0.5).exp()
}

/// Given 3 triangle edge lengths, compute the sine of the angle opposite to each edge.
pub fn sine_angles([a, b, c]: [F; 3]) -> [F; 3] {
    if a == 0. || b == 0. || c == 0. {
        return [0.; 3];
    }
    let area = herons_area([a, b, c]);
    // double circumradius
    let dbl_circumradius = a * b * c / (2. * area);
    [a, b, c].map(|v| v / dbl_circumradius)
}

#[test]
fn test_cosine_sine() {
    let tri = [[0.5, 0.1, -0.3], [-0.24, 0.3, -0.1], [0., 0.7, 0.2]];

    let [e0, e1, e2] = [
        sub(tri[1], tri[0]),
        sub(tri[2], tri[1]),
        sub(tri[0], tri[2]),
    ];

    let edge_lens = [length(e1), length(e2), length(e0)];

    use std::ops::Neg;
    let coss = cosine_angles(edge_lens);
    let [c0, c1, c2] = [
        [e0, e2.map(Neg::neg)],
        [e1, e0.map(Neg::neg)],
        [e2, e1.map(Neg::neg)],
    ]
    .map(|[a, b]| dot(normalize(a), normalize(b)));
    assert!((coss[0] - c0).abs() < 1e-5);
    assert!((coss[1] - c1).abs() < 1e-5);
    assert!((coss[2] - c2).abs() < 1e-5);

    let sines = sine_angles(edge_lens);
    let [s0, s1, s2] = [
        [e0, e2.map(Neg::neg)],
        [e1, e0.map(Neg::neg)],
        [e2, e1.map(Neg::neg)],
    ]
    .map(|[a, b]| length(cross(normalize(a), normalize(b))));
    assert!((sines[0] - s0).abs() < 1e-5);
    assert!((sines[1] - s1).abs() < 1e-5);
    assert!((sines[2] - s2).abs() < 1e-5);
}

/// More robust ln triangle area.
pub fn ln_tri_area<const N: usize>([a, b, c]: [[F; N]; 3]) -> F {
    let e0 = length(sub(a, b));
    let e1 = length(sub(b, c));
    let e2 = length(sub(c, a));
    let s = 0.5 * (e0 + e1 + e2);
    let acl = |s: F| (s.abs() + 1e-14).ln();
    0.5 * (acl(s) + acl(s - e0) + acl(s - e1) + acl(s - e2))
}

/// Given a triangle in 2d and a point p, compute the barycentric coordinate of point `p`.
pub fn barycentric_2d(p: [F; 2], [a, b, c]: [[F; 2]; 3]) -> [F; 3] {
    barycentric_n(p, a, b, c)
}

fn diff_of_prod(a: F, b: F, x: F, y: F) -> F {
    let xy = x * y;
    let dop = a.mul_add(b, -xy);
    let err = x.mul_add(-y, xy);
    dop + err
}

#[inline]
pub fn barycentric_n<const N: usize>(p: [F; N], a: [F; N], b: [F; N], c: [F; N]) -> [F; 3] {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = diff_of_prod(d00, d11, d01, d01);
    if denom.abs() < 1e-16 {
        return [1., 0., 0.];
    }
    let v = diff_of_prod(d11, d20, d01, d21) / denom;
    let w = diff_of_prod(d00, d21, d01, d20) / denom;
    [1.0 - v - w, v, w]
}

#[test]
fn test_bary_2d() {
    let tri = [[0., 0.], [1., 0.], [0., 1.]];

    assert_eq!(barycentric_2d([0., 0.], tri), [1., 0., 0.]);
    assert_eq!(barycentric_2d([1., 0.], tri), [0., 1., 0.]);
    assert_eq!(barycentric_2d([0., 1.], tri), [0., 0., 1.]);
}

pub fn barycentric_3d(p: [F; 3], [a, b, c]: [[F; 3]; 3]) -> [F; 3] {
    barycentric_n(p, a, b, c)
}

/// For a given direction in world space, convert it to a direction lying in the plane of the
/// triangle using basis defined by the barycentric coordinates.
pub fn dir_to_barycentric(dir: [F; 3], tri: [[F; 3]; 3]) -> [F; 2] {
    let p = add(tri[2], kmul(1e-5, dir));
    let [b0, b1, _] = barycentric_3d(p, tri);
    normalize([b0, b1])
}

/// Construct a 2D rotation matrix with rotation theta
pub(crate) fn rot_matrix_2d(theta: F) -> [[F; 2]; 2] {
    [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]]
}

pub fn matmul_2d([r0, r1]: [[F; 2]; 2], xy: [F; 2]) -> [F; 2] {
    [dot(r0, xy), dot(r1, xy)]
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

pub fn edges<T>(vis: &[T]) -> impl Iterator<Item = [T; 2]> + '_
where
    T: Copy,
{
    (0..vis.len()).map(|vi| [vis[vi], vis[(vi + 1) % vis.len()]])
}

// https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
fn tri_contains([v0, v1, v2]: [[F; 2]; 3], p: [F; 2]) -> bool {
    let d1 = cross_2d(sub(p, v1), sub(v0, v1));
    let d2 = cross_2d(sub(p, v2), sub(v1, v2));
    let d3 = cross_2d(sub(p, v0), sub(v2, v0));

    d1.is_sign_positive() == d2.is_sign_positive() && d1.is_sign_positive() == d3.is_sign_positive()
}

#[test]
fn test_tri_contains() {
    let t = [[0., 0.], [1., 0.], [0., 1.]];
    assert!(tri_contains(t, [0.25, 0.25]));
    assert!(tri_contains(t, [0.5, 0.5]));
    assert!(tri_contains(t, [1., 0.]));
    assert!(!tri_contains(t, [0.75, 0.75]));
    assert!(!tri_contains(t, [-0.5, 0.25]));
    assert!(!tri_contains(t, [0.25, -0.5]));
}

/// Computes the area of a polygon in 2D
fn poly_area_2d(p: &[[F; 2]]) -> F {
    let n = p.len();
    (0..n).map(|i| cross_2d(p[i], p[(i + 1) % n])).sum::<F>() / 2.
}
