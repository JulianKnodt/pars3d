use super::{F, add, cross, dot, kmul, normalize};

/// Computes the conjugate for inverse rotation of a quaternion.
#[inline]
pub fn conj([x, y, z, w]: [F; 4]) -> [F; 4] {
    [-x, -y, -z, w]
}

/// Multiplies two quaternions together
pub fn quat_mul([r1, r2, r3, r0]: [F; 4], [s1, s2, s3, s0]: [F; 4]) -> [F; 4] {
    [
        r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2,
        r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1,
        r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0,
        r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3,
    ]
}

#[inline]
pub fn quat_rot([x, y, z]: [F; 3], quat: [F; 4]) -> [F; 3] {
    let v = [x, y, z, 0.];
    let [a, b, c, _] = quat_mul(quat_mul(quat, v), conj(quat));
    [a, b, c]
}

pub fn orthogonal(v: [F; 3]) -> [F; 3] {
    assert!(v.iter().any(|&v| v != 0.));
    let [x, y, z] = v.map(F::abs);

    let other = if x <= y && x <= z {
        [1., 0., 0.]
    } else if y <= x && y <= z {
        [0., 1., 0.]
    } else {
        [0., 0., 1.]
    };
    cross(v, other)
}

/// Rodrigues' formula for axis-angle rotation
pub fn axis_angle_rot(k: [F; 3], v: [F; 3], angle: F) -> [F; 3] {
    let (s, c) = angle.sin_cos();
    add(
        kmul(c, v),
        add(kmul(s, cross(k, v)), kmul(dot(k, v) * (1. - c), k)),
    )
}

#[inline]
pub fn quat_from_to(s: [F; 3], t: [F; 3]) -> [F; 4] {
    let ns = normalize(s);
    let d = dot(ns, normalize(t));
    // opposite directions
    if d < -1. + 1e-5 {
        let [ox, oy, oz] = normalize(orthogonal(ns));
        return [ox, oy, oz, 0.];
    }

    let v = cross(t, s);
    normalize([v[0], v[1], v[2], 1. + d])
}

#[inline]
pub fn quat_from_axis_angle(axis: [F; 3], angle: F) -> [F; 4] {
    let s = (angle / 2.).sin();
    let [x, y, z] = axis.map(|v| v * s);
    [x, y, z, (angle / 2.).cos()]
}

/// Computes rotation from the standard xyz basis to this basis, where fwd and up are orthogonal
/// and normalized.
pub fn quat_from_standard(fwd: [F; 3], up: [F; 3]) -> [F; 4] {
    assert!(dot(fwd, up).abs() < 1e-4);
    let r0 = quat_from_to([1., 0., 0.], fwd);
    let r1 = quat_from_to(quat_rot([0., 1., 0.], r0), up);
    quat_mul(r1, r0)
}

pub fn quat_from_basis(fwd: [F; 3], up: [F; 3], b0: [F; 3], b1: [F; 3]) -> [F; 4] {
    assert!(dot(fwd, up).abs() < 1e-4);
    let r0 = quat_from_to(b0, fwd);
    let r1 = quat_from_to(quat_rot(b1, r0), up);
    quat_mul(r1, r0)
}

/// returns each row of the matrix representing a quaternion
pub fn quat_to_mat([x, y, z, w]: [F; 4]) -> [[F; 3]; 3] {
    let qxx = x * x;
    let qyy = y * y;
    let qzz = z * z;
    let qxz = x * z;
    let qxy = x * y;
    let qyz = y * z;
    let qwx = w * x;
    let qwy = w * y;
    let qwz = w * z;

    [
        [1. - 2. * (qyy + qzz), 2. * (qxy - qwz), 2. * (qxz + qwy)],
        [2. * (qxy + qwz), 1. - 2. * (qxx + qzz), 2. * (qyz - qwx)],
        [2. * (qxz - qwy), 2. * (qyz + qwx), 1. - 2. * (qxx + qyy)],
    ]
}

#[test]
fn test_quat() {
    let q = quat_from_to([1., 0., 0.], [0., 1., 0.]);
    let rot = quat_rot([1., 0., 0.], q);
    assert!(super::dist(rot, [0., 1., 0.]) < 1e-3);
}

#[test]
fn test_quat_from_to_parallel() {
    let e0 = [1., 0., 0.];
    let rot = quat_from_to(e0, e0);
    assert_eq!(quat_rot(e0, rot), e0);
    use core::ops::Neg;
    let neg_e0 = e0.map(Neg::neg);
    let opp_rot = quat_from_to(e0, neg_e0);
    assert_ne!(super::length(opp_rot), 0.);
    assert_eq!(quat_rot(e0, opp_rot), neg_e0, "{opp_rot:?}");
}

#[test]
pub fn test_quat_basis() {
    let tgt = normalize([0., 0.5, 0.5]);
    let up = [1., 0., 0.];

    let q = quat_from_standard(tgt, up);

    let r0 = quat_rot([1., 0., 0.], q);
    let r1 = quat_rot([0., 1., 0.], q);
    assert!(super::dist(r0, tgt) < 1e-4);
    assert!(super::dist(r1, up) < 1e-4);
}

#[test]
fn test_quat_from_standard() {
    let fwd = [0., 1., 0.];
    let up = [1., 0., 0.];

    let q = quat_from_standard(fwd, up);
    assert!((super::length(q) - 1.).abs() < 1e-4, "{q:?}");
}

#[test]
fn test_identity_quat() {
    let n = normalize([1.; 3]);
    let q = quat_from_to(n, n);
    assert_eq!(quat_rot(n, q), n);
}
