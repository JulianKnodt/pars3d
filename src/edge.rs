/// Description of an edge and its adjacent faces
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind<T = usize> {
    Boundary(T),
    Manifold([T; 2]),
    NonManifold(Vec<T>),
}

const _: () = assert!(
    std::mem::size_of::<EdgeKind>() == 24,
    "Size of EdgeKind is too large",
);

impl<T> EdgeKind<T> {
    pub fn as_slice(&self) -> &[T] {
        match self {
            EdgeKind::Boundary(f) => std::slice::from_ref(f),
            EdgeKind::Manifold(fs) => fs.as_slice(),
            EdgeKind::NonManifold(fs) => fs.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            EdgeKind::Boundary(f) => std::slice::from_mut(f),
            EdgeKind::Manifold(fs) => fs.as_mut_slice(),
            EdgeKind::NonManifold(fs) => fs.as_mut_slice(),
        }
    }

    pub fn empty() -> Self {
        EdgeKind::NonManifold(vec![])
    }
    pub fn len(&self) -> usize {
        match self {
            EdgeKind::Boundary(_) => 1,
            EdgeKind::Manifold(_) => 2,
            EdgeKind::NonManifold(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            EdgeKind::NonManifold(v) => v.is_empty(),
            _ => false,
        }
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
}

impl<T> EdgeKind<(usize, T)> {
    /// Insert a new element into this edge kind, if a condition is met for all existing faces.
    pub fn insert_if_new(&mut self, fi: usize, new: impl Fn() -> T) -> bool {
        macro_rules! nt {
            () => {
                (fi, new())
            };
        }
        let prev = std::mem::replace(self, Self::empty());
        let new = match prev {
            EdgeKind::Boundary(f) if f.0 != fi => EdgeKind::Manifold([f, nt!()]),
            EdgeKind::Manifold([f0, f1]) if f0.0 != fi && f1.0 != fi => {
                EdgeKind::NonManifold(vec![f0, f1, nt!()])
            }
            EdgeKind::NonManifold(mut fs) if fs.iter().all(|p| p.0 != fi) => match fs.len() {
                0 => EdgeKind::Boundary(nt!()),
                1 => EdgeKind::Manifold([fs.pop().unwrap(), nt!()]),
                _ => {
                    fs.push(nt!());
                    EdgeKind::NonManifold(fs)
                }
            },
            _ => {
                *self = prev;
                return false;
            }
        };
        *self = new;
        true
    }

    pub fn get_mut_or_insert_with(&mut self, fi: usize, new: impl Fn() -> T) -> &mut T {
        self.insert_if_new(fi, new);
        unsafe {
            &mut self
                .as_mut_slice()
                .iter_mut()
                .find(|f| f.0 == fi)
                .unwrap_unchecked()
                .1
        }
    }
}

impl EdgeKind {
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
