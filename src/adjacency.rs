use super::Mesh;
use std::collections::BTreeMap;

/// Structure for maintaining vertex adjacencies of a mesh with fixed topology.
/// If the mesh is modified, this structure will no longer be valid.
#[derive(Debug, Clone)]
pub struct VertexAdj {
    /// Index and # of nbrs in adjacency vector
    idx_count: Vec<(u32, u16)>,
    /// Flattened 2D vector from vertex to its neighbors
    adj: Vec<u32>,
}

impl Mesh {
    pub fn vertex_adj(&self) -> VertexAdj {
        let mut nbrs: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for f in &self.f {
            for [e0, e1] in f.edges() {
                nbrs.entry(e0).or_default().push(e1 as u32);
                nbrs.entry(e1).or_default().push(e0 as u32);
            }
        }
        let mut idx_count = vec![];
        let mut adj = vec![];
        for vi in 0..self.v.len() {
            let Some(n) = nbrs.get_mut(&vi) else {
                idx_count.push((0, 0));
                continue;
            };
            assert!(n.len() < u16::MAX as usize);
            idx_count.push((adj.len() as u32, n.len() as u16));
            adj.append(n);
        }

        VertexAdj { idx_count, adj }
    }
}

impl VertexAdj {
    /// The adjacent vertices to this index.
    pub fn adj(&self, v: usize) -> &[u32] {
        let (idx, cnt) = self.idx_count[v];
        if cnt == 0 {
            return &[];
        }
        let idx = idx as usize;
        let cnt = cnt as usize;

        &self.adj[idx..idx + cnt]
    }
}
