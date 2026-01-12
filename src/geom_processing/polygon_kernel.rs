use crate::isect::line_isect;
use crate::{F, Sign, orient};

/// https://dl.acm.org/doi/pdf/10.1145/322139.322142
pub fn polygon_kernel(vs: &[[F; 2]], out: &mut Vec<[F; 2]>) {
    out.clear();
    out.extend(vs.iter().copied());

    #[allow(non_snake_case)]
    let N = vs.len();
    for vi in 0..N {
        let v = vs[vi];
        let vn = vs[(vi + 1) % N];
        let vp = vs[(vi + N - 1) % N];
        if orient([v, vn, vp]) != Sign::Neg {
            continue;
        }

        // TODO can hoist this buffer to parameters?
        let mut out_buf = vec![];
        let mut ki = 0;
        while ki < out.len() {
            let k = out[ki];
            let kn = out[(ki + 1) % out.len()];

            let t1 = orient([vp, v, k]);
            let t2 = orient([vp, v, kn]);
            let t3 = orient([v, vn, k]);
            let t4 = orient([v, vn, kn]);
            if !(t1 == Sign::Neg || t3 == Sign::Neg) {
                out_buf.push(k);
            }
            if t1 != t2 && t1 != Sign::Zero && t2 != Sign::Zero {
                let (_, _, isect) = line_isect([vp, v], [k, kn]).unwrap();
                out_buf.push(isect);
            }
            if t3 != t4 && t3 != Sign::Zero && t4 != Sign::Zero {
                let (_, _, isect) = line_isect([v, vn], [k, kn]).unwrap();
                out_buf.push(isect);
            }

            ki += 1;
        }
        std::mem::swap(out, &mut out_buf);
        out_buf.clear();
    }
}

/// Computes the kernel for a given quadrilateral.
/// assumes that vs are in counterclockwise order
pub fn quad_kernel(vs: [[F; 2]; 4]) -> [[F; 2]; 4] {
    use crate::{cross_2d, sub};
    use std::array::from_fn;
    from_fn(|i| {
        let [v, v_n, v_nn, v_nnn] = from_fn(|j| vs[(i + j) % 4]);
        let isect = if cross_2d(sub(v_nn, v_n), sub(v, v_n)) < 0. {
            line_isect([v_nnn, v_nn], [v, v_n])
        } else if cross_2d(sub(v_nnn, v_nn), sub(v, v_nn)) < 0. {
            line_isect([v_n, v_nn], [v_nnn, v])
        } else {
            return v;
        };
        isect.unwrap().2
    })
}

#[test]
fn test_quad_polygon_kernel() {
    let verts = &[
        [1., 0.], //
        [-3., 1.],
        [-1., 0.],
        [-3., -1.],
    ];
    let mut out = vec![];
    polygon_kernel(verts, &mut out);
    assert_eq!(out.len(), 4);
}

#[test]
fn test_quad_quad_kernel() {
    let verts = [
        [1., 0.], //
        [-3., 1.],
        [-1., 0.],
        [-3., -1.],
    ];
    let out = quad_kernel(verts);

    let mut out_poly = vec![];
    polygon_kernel(&verts, &mut out_poly);
    // cw vs ccw (probably should make it consistent
    out_poly.swap(1, 3);
    assert_eq!(out_poly.len(), out.len());
    for i in 0..4 {
        assert!(
            crate::dist(out[i], out_poly[i]) < 1e-6,
            "{:?} {:?}",
            out[i],
            out_poly[i]
        );
    }
}

#[test]
fn test_degen_quad_kernel() {
    let verts = [[1., 0.], [1. + 1e-8, 0.], [0., 1.], [0., -1.]];
    let out = quad_kernel(verts);
    assert_eq!(out, [[1., 0.], [1., 0.], [0., 1.], [0., -1.],]);
}

#[test]
fn test_complex_concave() {
    let verts = &[
        [1., 2.],
        [0., 1.],
        [-1., 3.],
        [-1., -3.],
        [0., -1.],
        [1., -3.],
    ];

    let mut out = vec![];
    polygon_kernel(verts, &mut out);
    assert_eq!(out.len(), 4);
}
