use super::barycentric_areas;
use crate::{dot, normalize, sub, FaceKind, F};

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
