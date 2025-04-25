use super::Mesh;
use std::collections::{BTreeMap, BTreeSet};

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
                assert_ne!(e0, e1);
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

    /// Connectivity between boundary vertices (vert -> [prev, next])
    /// Also returns the number of boundary loops present in this mesh.
    pub fn boundary_loops(&self, m: &Mesh) -> (usize, BTreeMap<usize, [usize; 2]>) {
        let mut bd_verts = m.boundary_vertices().collect::<BTreeSet<_>>();

        let mut out = BTreeMap::new();
        let mut num_loops = 0;
        while let Some(s) = bd_verts.pop_first() {
            num_loops += 1;
            let nbrs = self.adj(s);
            let next = *nbrs
                .iter()
                .find(|&&v| bd_verts.contains(&(v as usize)))
                .unwrap();
            let mut prev = *nbrs
                .iter()
                .find(|&&v| bd_verts.contains(&(v as usize)) && v != next)
                .unwrap() as usize;
            let next = next as usize;
            assert_eq!(out.insert(s as usize, [prev, next]), None);
            assert_eq!(out.insert(next, [s, usize::MAX]), None);
            assert_eq!(out.insert(prev, [usize::MAX, s]), None);
            prev = s;
            let mut curr = next;
            loop {
                assert!(bd_verts.remove(&curr));
                let nbrs = self.adj(curr);
                let Some(&next) = nbrs.iter().find(|&&v| bd_verts.contains(&(v as usize))) else {
                    break;
                };
                let next = next as usize;

                assert_ne!(next, prev);
                out.entry(curr).or_insert([usize::MAX; 2])[1] = next;
                out.entry(next)
                    .and_modify(|v| {
                        v[0] = curr;
                    })
                    .or_insert([curr, usize::MAX]);
                prev = curr;
                curr = next;
            }
        }
        assert!(out
            .values()
            .all(|&[v0, v1]| v0 != usize::MAX && v1 != usize::MAX));

        (num_loops, out)
    }
}
