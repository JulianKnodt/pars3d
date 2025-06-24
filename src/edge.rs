/// Description of an edge and its adjacent faces
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    Boundary(usize),
    Manifold([usize; 2]),
    NonManifold(Vec<usize>),
}

impl EdgeKind {
    pub fn as_slice(&self) -> &[usize] {
        match self {
            EdgeKind::Boundary(f) => std::slice::from_ref(f),
            EdgeKind::Manifold(fs) => fs.as_slice(),
            EdgeKind::NonManifold(fs) => fs.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            EdgeKind::Boundary(f) => std::slice::from_mut(f),
            EdgeKind::Manifold(fs) => fs.as_mut_slice(),
            EdgeKind::NonManifold(fs) => fs.as_mut_slice(),
        }
    }
    pub fn insert(&mut self, v: usize) -> bool {
        let new = match self {
            &mut EdgeKind::Boundary(f) if f != v => EdgeKind::Manifold(std::cmp::minmax(f, v)),
            &mut EdgeKind::Manifold([f0, f1]) if f0 != v && f1 != v => {
                EdgeKind::NonManifold(vec![f0, f1, v])
            }
            EdgeKind::NonManifold(fs) if !fs.contains(&v) => match fs.len() {
                0 => EdgeKind::Boundary(v),
                1 => EdgeKind::Manifold([fs[0], v]),
                _ => {
                    fs.push(v);
                    return true;
                }
            },
            _ => return false,
        };
        *self = new;
        true
    }
    pub(crate) fn empty() -> Self {
        EdgeKind::NonManifold(vec![])
    }
    pub fn is_boundary(&self) -> bool {
        matches!(self, EdgeKind::Boundary(_))
    }
    pub fn is_non_manifold(&self) -> bool {
        matches!(self, EdgeKind::NonManifold(_))
    }
    pub fn is_manifold(&self) -> bool {
        matches!(self, EdgeKind::Manifold(_))
    }
    /// Returns the element other than `i` if self is `Manifold`, otherwise returns `None`.
    /// If `i` is not in the `Manifold` edge then returns `None`.
    #[inline]
    pub fn opposite(&self, i: usize) -> Option<usize> {
        match self {
            &EdgeKind::Manifold([a, b] | [b, a]) if a == i => Some(b),
            _ => None,
        }
    }
}
