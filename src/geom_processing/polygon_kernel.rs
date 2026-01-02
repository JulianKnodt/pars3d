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
