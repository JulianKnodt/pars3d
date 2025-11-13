use super::{F, cross, dot, sub};

fn stp(a: [F; 3], b: [F; 3], c: [F; 3]) -> F {
    dot(a, cross(b, c))
}

// https://stackoverflow.com/questions/38545520/barycentric-coordinates-of-a-tetrahedron

pub fn bary_tet(a: [F; 3], b: [F; 3], c: [F; 3], d: [F; 3], p: [F; 3]) -> [F; 4] {
    let pa = sub(p, a);
    let pb = sub(p, b);

    let ba = sub(b, a);
    let ca = sub(c, a);
    let da = sub(d, a);

    let cb = sub(c, b);
    let db = sub(d, b);
    let va6 = stp(pb, db, cb);
    let vb6 = stp(pa, ca, da);
    let vc6 = stp(pa, da, ba);
    let vd6 = stp(pa, ba, ca);
    let v6 = 1. / stp(ba, ca, da);

    [va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6]
}
