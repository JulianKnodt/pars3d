use super::F;

pub enum FaceKind {
  Tri([usize; 3]),
  Quad([usize; 4]),
  Poly(Vec<usize>),
}

struct Mesh {
  v: Vec<[F; 3]>,
  f: Vec<FaceKind>,
}
