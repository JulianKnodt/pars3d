use super::{F, Vec3};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use super::I;

/// Returns mapping between original vertex coordinate and new vertex coordinate
pub fn fuse_vertices(vertices: &[Vec3], dist: F) -> Vec<usize> {
    let mut out = (0..vertices.len()).collect::<Vec<_>>();
    if dist < 0. {
        return out;
    }
    let mut spatial_hash = HashMap::new();
    let inv_dist = dist.recip();

    let to_hash = |v: Vec3| {
        if dist == 0. || inv_dist.is_nan() {
            return v.map(|v| v.to_bits().cast_signed());
        }

        v.map(|v| f64::from(v * inv_dist) as I)
    };

    for (i, v) in vertices.iter().enumerate() {
        let h = to_hash(*v);
        match spatial_hash.entry(h) {
            Entry::Occupied(o) => {
                out[i] = *o.get();
            }
            Entry::Vacant(v) => {
                v.insert(i);
            }
        }
    }
    out
}

/// Given a remapping, set the vertices to be exactly overlapped on top of each other.
#[inline]
pub fn set_equal(remap: &[usize], verts: &mut [[F; 3]]) {
    for (dst_vi, &src_vi) in remap.iter().enumerate() {
        verts[dst_vi] = verts[src_vi];
    }
}

/// Applies a mapping to a set of indices.
#[inline]
pub fn apply_fuse<'a>(remap: &[usize], to: impl Iterator<Item = &'a mut usize>) {
    for vi in to {
        *vi = remap[*vi];
    }
}

#[test]
fn test_fuse() {
    let a = [0., 0., 0.];
    let b = [0., 0., 1e-5];
    assert_eq!(fuse_vertices(&[a, b], 0.), vec![0, 1]);
    assert_eq!(fuse_vertices(&[a, b], 1e-4), vec![0, 0]);

    assert_eq!(fuse_vertices(&[a, a], 0.), vec![0, 0]);
    assert_eq!(fuse_vertices(&[a, a], 1e-8), vec![0, 0]);
}
