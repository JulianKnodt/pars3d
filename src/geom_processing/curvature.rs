use super::barycentric_areas;
use crate::{F, FaceKind, add, dot, kmul, normalize, sub, tri_area};

/// Compute gaussian curvature at each vertex.
pub fn gaussian_curvature(fs: &[FaceKind], vs: &[[F; 3]], dst: &mut Vec<F>) {
    dst.fill(0.);
    dst.resize(vs.len(), 0.);
    for f in fs {
        for pcn in f.incident_edges() {
            let [prev, curr, next] = pcn.map(|vi| vs[vi]);
            let e0 = normalize(sub(prev, curr));
            let e1 = normalize(sub(next, curr));
            let cos_ang = dot(e0, e1).clamp(-1., 1.);
            dst[pcn[1]] -= cos_ang.acos();
        }
    }
    let mut buf = vec![std::f64::consts::TAU as F; vs.len()];
    barycentric_areas(fs, vs, &mut buf);
    assert_eq!(buf.len(), dst.len());
    for vi in 0..buf.len() {
        if buf[vi] == 0. {
            // not sure what to do if there is no area here
            dst[vi] = 0.;
        } else {
            dst[vi] /= buf[vi];
        }
    }
}

/// Compute the mean curvature at each vertex
pub fn mean_curvature(fs: &[FaceKind], vs: &[[F; 3]], dst: &mut Vec<[F; 3]>) {
    dst.resize(vs.len(), [0.; 3]);
    let mut bary_areas = vec![0.; vs.len()];
    for f in fs.iter() {
        for pcn in f.incident_edges() {
            let [p, c, n] = pcn.map(|vi| vs[vi]);
            let [pi, ci, ni] = pcn;

            let area = tri_area([p, c, n]) / 3.;
            for i in pcn {
                bary_areas[i] += area;
            }
            let pc = normalize(sub(p, c));
            let pn = normalize(sub(p, n));
            let cn = normalize(sub(c, n));
            let p_cot = dot(pc, pn).acos().tan().recip();
            let n_cot = dot(cn, pn).acos().tan().recip();
            let c_cot = dot(pc, cn.map(|v| -v)).acos().tan().recip();

            dst[ci] = add(dst[ci], kmul(p_cot / 2., sub(n, c)));
            dst[ni] = add(dst[ni], kmul(p_cot / 2., sub(c, n)));

            dst[ci] = add(dst[ci], kmul(n_cot / 2., sub(p, c)));
            dst[pi] = add(dst[pi], kmul(n_cot / 2., sub(c, p)));

            dst[ni] = add(dst[ni], kmul(c_cot / 2., sub(p, n)));
            dst[pi] = add(dst[pi], kmul(c_cot / 2., sub(n, p)));
        }
    }
}

pub fn mean_curvature_signed_norm(vns: &[[F; 3]], mean_curvature: &[[F; 3]], dst: &mut Vec<F>) {
    assert_eq!(mean_curvature.len(), vns.len());

    let nv = vns.len();
    dst.resize(nv, 0.);
    for i in 0..nv {
        let sign = dot(vns[i], mean_curvature[i]).signum();
        dst[i] = sign * crate::length(mean_curvature[i]);
    }
}
