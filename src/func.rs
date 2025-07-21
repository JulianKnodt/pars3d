use super::{F, add, kmul};

/// How to define color on a curve
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarFn<const N: usize> {
    /// Constant color on the whole curve
    Constant([F; N]),

    /// Linearly interpolate the color along the curve
    Linear([F; N], [F; N]),
    // TODO add more complex functions here?
}

impl<const N: usize> ScalarFn<N> {
    pub fn start(&self) -> [F; N] {
        use ScalarFn::*;
        match *self {
            Constant(c) => c,
            Linear(s, _) => s,
        }
    }
    pub fn end(&self) -> [F; N] {
        use ScalarFn::*;
        match *self {
            Constant(c) => c,
            Linear(_, e) => e,
        }
    }
    pub fn lerp(&self, t: F) -> [F; N] {
        use ScalarFn::*;
        match *self {
            Constant(c) => c,
            Linear(s, e) => add(kmul(1. - t, s), kmul(t, e)),
        }
    }
    pub fn split(&self) -> [Self; 2] {
        use ScalarFn::*;
        match *self {
            Constant(c) => [Constant(c); 2],
            Linear(s, e) => {
                let mid = self.lerp(0.5);
                [Linear(s, mid), Linear(mid, e)]
            }
        }
    }
    pub fn reverse(&self) -> Self {
        use ScalarFn::*;
        match *self {
            Constant(c) => Constant(c),
            Linear(s, e) => Linear(e, s),
        }
    }
}

impl Default for ScalarFn<3> {
    fn default() -> Self {
        Self::Constant([0.; 3])
    }
}
