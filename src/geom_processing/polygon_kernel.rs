use crate::isect::line_isect;
use crate::{F, cross_2d, sub};

pub fn polygon_kernel(vs: &[[F; 2]], out: &mut Vec<[F; 2]>) {
    out.clear();
    out.extend(vs.iter().copied());
    let orient = |a, b, c| sign(cross_2d(sub(a, c), sub(b, c)));

    #[allow(non_snake_case)]
    let N = vs.len();
    for vi in 0..N {
        let v = vs[vi];
        let vn = vs[(vi + 1) % N];
        let vp = vs[(vi + N - 1) % N];
        if orient(v,vn, vp) != Sign::Neg {
            continue;
        }

        let mut out_buf = vec![];
        let mut ki = 0;
        while ki < out.len() {
            let k = out[ki];
            let kn = out[(ki + 1) % out.len()];

            let t1 = orient(vp, v, k);
            let t2 = orient(vp, v, kn);
            let t3 = orient(v, vn, k);
            let t4 = orient(v, vn, kn);
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

            ki = ki + 1;
        }
        std::mem::swap(out, &mut out_buf);
        out_buf.clear();
    }
}
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
enum Sign {
  Pos,
  Neg,
  Zero
}
fn sign(x: F) -> Sign {
  if x > 0. {
    Sign::Pos
  } else if x < 0. {
    Sign::Neg
  } else {
    Sign::Zero
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
    for [x,y] in verts {
      print!("({x}, {y}), ");
    }
    println!();
    let mut out = vec![];
    polygon_kernel(verts, &mut out);
    assert_eq!(out.len(), 4);
    for [x,y] in out {
      print!("({x}, {y}), ");
    }
}

/*
#[test]
fn test_complex_concave
*/
