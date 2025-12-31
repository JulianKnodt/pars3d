use super::*;

// given two lines, compute their intersection
// Returns the length along each line which the intersection occurs
pub fn line_isect([a0, a1]: [[F; 2]; 2], [b0, b1]: [[F; 2]; 2]) -> Option<(F, F, [F; 2])> {
    // each point could also lie directly on the line, but that's annoying to check
    if a0 == a1 || b0 == b1 {
        return None;
    }
    let a_dir = sub(a1, a0);
    let b_dir = sub(b1, b0);

    let denom = cross_2d(a_dir, b_dir);
    if denom == 0. {
        // they're parallel
        return None;
    }
    let t = cross_2d(sub(b0, a0), b_dir) / denom;
    let u = cross_2d(sub(b0, a0), a_dir) / denom;

    Some((t, u, add(a0, kmul(t, a_dir))))
}

pub fn line_dir_isect([a0, a_dir]: [[F; 2]; 2], [b0, b_dir]: [[F; 2]; 2]) -> Option<[F; 2]> {
    let denom = cross_2d(a_dir, b_dir);
    if denom == 0. {
        return None;
    }
    let t = cross_2d(sub(b0, a0), b_dir) / denom;
    Some(add(a0, kmul(t, a_dir)))
}

/// Checks if point is on a line segment.
pub fn point_on_line_segment([s, e]: [[F; 2]; 2], p: [F; 2]) -> bool {
    (dist(s, p) + dist(e, p) - dist(s, e)).abs() < F::EPSILON
}

// given two lines, compute their intersection
pub fn line_segment_isect(a: [[F; 2]; 2], b: [[F; 2]; 2]) -> Option<[F; 2]> {
    // they could also lie directly on the line, but that's annoying to check
    let (t, u, isect) = line_isect(a, b)?;
    let valid = (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u);
    let pls = point_on_line_segment;
    let valid = valid && !pls(a, b[0]) && !pls(a, b[1]) && !pls(b, a[0]) && !pls(b, a[1]);
    valid.then_some(isect)
}

#[test]
fn test_line_segment_isect() {
    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0., 1.], [0., -1.]]);
    assert_eq!(Some([0., 0.]), isect);
    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0., -1.], [0., 1.]]);
    assert_eq!(Some([0., 0.]), isect);

    let isect = line_segment_isect([[-1., 0.], [1., 0.]], [[0.5, 100.], [-0.5, -100.]]);
    assert_eq!(Some([0., 0.]), isect);
}

/// Given the equation of a plane (dot(P,N) + d), where d = (dot(Point on Plane, N))
/// And some ray, compute the intersection of the ray and the plane.
/// If the ray and the plane are parallel, will return NaN first value.
pub fn line_plane_isect((n, d): ([F; 3], F), [o, dir]: [[F; 3]; 2]) -> (F, [F; 3]) {
    let t = -(dot(o, n) + d) / dot(dir, n);
    (t, add(o, kmul(t, dir)))
}

/// Given a triangle, computes the coefficients for the plane that passes through the triangle
pub fn plane_eq([a, b, c]: [[F; 3]; 3]) -> ([F; 3], F) {
    let n = cross(sub(c, a), sub(b, a));
    let n = normalize(n);
    (n, -dot(n, a))
}

pub fn dist_to_plane_eq((n, d): ([F; 3], F), p: [F; 3]) -> F {
    dot(n, p) + d
}

pub fn dist_to_plane(tri: [[F; 3]; 3], p: [F; 3]) -> F {
    dist_to_plane_eq(plane_eq(tri), p)
}

#[test]
fn test_line_plane_isect() {
    let p = ([0., 1., 0.], 0.);
    let (t, pos) = line_plane_isect(p, [[0., 1., 0.], [0., -1., 0.]]);
    assert_eq!(pos, [0., 0., 0.]);
    assert_eq!(t, 1.);

    let p = ([0., -1., 0.], 0.);
    let (t, pos) = line_plane_isect(p, [[0., 1., 0.], [0., -1., 0.]]);
    assert_eq!(pos, [0., 0., 0.]);
    assert_eq!(t, 1.);

    let (t, pos) = line_plane_isect(p, [[0.5, 0.5, 0.], [-0.5, -0.5, 0.]]);
    assert_eq!(pos, [0., 0., 0.]);
    assert_eq!(t, 1.);
}

#[test]
fn test_line_plane_isect_tri() {
    let tri = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let pl = plane_eq([tri[0], tri[1], tri[2]]);
    println!("{pl:?}");
    let ray = [[-1.0, 3.0, 0.0], [1.0, -4.0, 0.0]];
    assert_eq!(line_plane_isect(pl, ray).1, line_tri_isect(tri, ray).3);
}

/// Intersect a ray with a triangle.
/// Returns the barycentric coordinate in the first two values,
/// Returns ray distance in the third, and position in the last.
pub fn line_tri_isect([v0, v1, v2]: [[F; 3]; 3], [o, d]: [[F; 3]; 2]) -> (F, F, F, [F; 3]) {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let h = cross(d, e2);
    let a = dot(e1, h);
    //println!("a {a:?}");
    /*
    if -EPS < a && a < EPS {
      // parallel ray
      return None;
    }
    */
    let f = a.recip();
    let s = sub(o, v0);
    let u = f * dot(s, h);
    /*
    if u < 0. || u > 1. {
      return None;
    }
    */
    let q = cross(s, e1);
    let v = f * dot(d, q);
    /*
    if u < 0. || u + v > 1. {
      return None;
    }
    */
    let t = f * dot(e2, q);
    /*
    if t < EPS {
      return None;
    }
    */
    let pos = add(o, kmul(t, d));
    (u, v, t, pos)
}

#[test]
fn test_line_tri_isect() {
    let tri = [[-1., -1., 0.], [0., 1., 0.], [1., -1., 0.]];
    let o = [0., 0., -1.];
    let d = [0., 0., 1.];
    let (u, v, t, pos) = line_tri_isect(tri, [o, d]);
    println!("{u} {v} {t} {pos:?}");
    assert!(t > 0.);
    assert!((0.0..=1.0).contains(&u));
    assert!((0.0..=1.0).contains(&(u + v)));

    let tri = [[-1., -1., 0.], [0., 1., 0.], [1., -1., 0.]];
    let o = [5., 5., -1.];
    let d = [5., 5., 1.];
    let (u, v, t, pos) = line_tri_isect(tri, [o, d]);
    assert!(!(0.0..=1.0).contains(&u));
    assert!(!(0.0..=1.0).contains(&(u + v)));
    assert!(t > 0.);
    assert!(pos[2] == 0.);

    let tri = [[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]];
    let o = [0., 0., -1.];
    let d = [0., 0., 1.];
    let (u, v, t, pos) = line_tri_isect(tri, [o, d]);
    println!("{u} {v} {t} {pos:?}");
    assert!(t > 0.);
    assert!((0.0..=1.0).contains(&u));
    assert!((0.0..=1.0).contains(&(u + v)));
}

#[test]
fn test_line_tri_isect_on_tri_edge() {
    let o = [-1., 3., 0.];
    let at = [0., 1., 0.];

    let tri = [[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]];
    let (u, v, t, _pos) = line_tri_isect(tri, [o, sub(at, o)]);
    assert!((0.0..=1.0).contains(&(u + v)));
    assert!(t >= 0.);
    //println!("{u} {v} {t} {_pos:?}");
}
