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
    pub fn insert(&mut self, v: usize) -> bool {
        let new = match self {
            EdgeKind::Boundary(f) if *f != v => EdgeKind::Manifold([*f, v]),
            EdgeKind::Manifold(fs) if !fs.contains(&v) => {
                EdgeKind::NonManifold(vec![fs[0], fs[1], v])
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
    pub fn is_boundary(&self) -> bool {
        matches!(self, EdgeKind::Boundary(_))
    }
    pub fn is_nonmanifold(&self) -> bool {
        matches!(self, EdgeKind::NonManifold(_))
    }
    pub fn is_manifold(&self) -> bool {
        matches!(self, EdgeKind::Manifold(_))
    }
}
