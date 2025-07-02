use crate::{Mesh, F};
use kdtree::KDTree;

impl Mesh {
    /// Constructs a KDTree for this mesh.
    /// Stores for each tri in the mesh its original face, along with its tri index.
    pub fn kdtree(&self) -> KDTree<(usize, u32), 3, 3, F> {
        let pts = self.f.iter().enumerate().flat_map(|(fi, f)| {
            f.as_triangle_fan()
                .enumerate()
                .map(move |(ti, ijk)| (ijk.map(|vi| self.v[vi]), (fi, ti as u32)))
        });
        KDTree::<_, _, _, F>::new(pts, Default::default())
    }
}
