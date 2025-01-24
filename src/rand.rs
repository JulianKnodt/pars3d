use super::F;

// quasi random generation of floats in the range [0,1)
#[inline]
pub fn quasi_rand(i: usize) -> F {
    // plastic constant?
    const G: F = 1.6180339887498948482;
    const A1: F = 1. / G;
    (0.5 + A1 * (i as F)).fract()
}

#[test]
fn test_rand_range() {
    for i in 0..100 {
        let v = quasi_rand(i);
        assert!((0.0..=1.0).contains(&v), "{v}");
    }
}
