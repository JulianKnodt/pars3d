use super::Mesh;
use std::collections::{BTreeMap, BTreeSet};

/// Structure for maintaining vertex adjacencies of a mesh with fixed topology.
/// If the mesh is modified, this structure will no longer be valid.
/// Can also be used to store associated data.
#[derive(Debug, Clone)]
pub struct VertexAdj<D = ()> {
    /// Index and # of nbrs in adjacency vector
    idx_count: Vec<(u32, u16)>,
    /// Flattened 2D vector from vertex to its neighbors
    adj: Vec<u32>,
    /// Associated data with each edge
    data: Vec<D>,
}

impl Mesh {
    pub fn vertex_adj(&self) -> VertexAdj<()> {
        let mut nbrs: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for f in &self.f {
            for [e0, e1] in f.edges() {
                assert_ne!(e0, e1);
                let e0_nbrs = nbrs.entry(e0).or_default();
                if !e0_nbrs.contains(&(e1 as u32)) {
                    e0_nbrs.push(e1 as u32);
                }
                let e1_nbrs = nbrs.entry(e1).or_default();
                if !e1_nbrs.contains(&(e0 as u32)) {
                    e1_nbrs.push(e0 as u32);
                }
            }
        }
        let mut idx_count = vec![];
        let mut adj = vec![];
        for vi in 0..self.v.len() {
            let Some(n) = nbrs.get_mut(&vi) else {
                // no neighbors
                idx_count.push((0, 0));
                continue;
            };
            assert!(n.len() < u16::MAX as usize);
            idx_count.push((adj.len() as u32, n.len() as u16));
            adj.append(n);
        }

        let data = vec![(); adj.len()];
        VertexAdj {
            idx_count,
            adj,
            data,
        }
    }
}

impl<D> VertexAdj<D> {
    /// Modifies the data for this vertex adjacency based on a function which takes an ordered
    /// edge.
    pub fn map<U>(self, f: impl Fn(&Self, usize, usize, D) -> U) -> VertexAdj<U>
    where
        U: Default + Copy,
        D: Copy,
    {
        let mut data = vec![U::default(); self.data.len()];
        for ([v0, v1], idx) in self.all_pairs_with_idx() {
            data[idx] = f(&self, v0, v1, self.data[idx]);
        }

        let Self {
            data: _,
            adj,
            idx_count,
        } = self;

        VertexAdj {
            data,
            adj,
            idx_count,
        }
    }

    /// Returns the degree of a given vertex
    pub fn degree(&self, v: usize) -> usize {
        self.idx_count[v].1 as usize
    }
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

    pub fn data(&self, v: usize) -> &[D] {
        let (idx, cnt) = self.idx_count[v];
        if cnt == 0 {
            return &[];
        }
        let idx = idx as usize;
        let cnt = cnt as usize;

        &self.data[idx..idx + cnt]
    }

    pub fn adj_data(&self, v: usize) -> impl Iterator<Item = (u32, D)> + '_
    where
        D: Copy,
    {
        let (idx, cnt) = self.idx_count[v];
        (idx..idx + cnt as u32).map(|i| {
            let i = i as usize;
            unsafe { (*self.adj.get_unchecked(i), *self.data.get_unchecked(i)) }
        })
    }

    pub fn adj_data_mut(&mut self, v: usize) -> (&[u32], &mut [D]) {
        let (idx, cnt) = self.idx_count[v];
        if cnt == 0 {
            return (&[], &mut []);
        }
        let idx = idx as usize;
        let cnt = cnt as usize;

        (&self.adj[idx..idx + cnt], &mut self.data[idx..idx + cnt])
    }

    /// Returns all pairs of edges (both e0->e1 and e1->e0)
    pub fn all_pairs(&self) -> impl Iterator<Item = ([usize; 2], D)> + '_
    where
        D: Copy,
    {
        self.all_pairs_with_idx()
            .map(|(v, idx)| (v, self.data[idx]))
    }

    fn all_pairs_with_idx(&self) -> impl Iterator<Item = ([usize; 2], usize)> + '_ {
        self.idx_count
            .iter()
            .enumerate()
            .flat_map(move |(v0, &(idx, cnt))| {
                (idx..idx + cnt as u32)
                    .map(move |i| ([v0, self.adj[i as usize] as usize], i as usize))
            })
    }

    /// Returns all pairs of edges once, such that e0 < e1
    pub fn all_pairs_ord(&self) -> impl Iterator<Item = ([usize; 2], D)> + '_
    where
        D: Copy,
    {
        self.all_pairs().filter(|([e0, e1], _)| e0 < e1)
    }

    // TODO this should return another struct which also contains the vertices enclosed within
    // each boundary loop.
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
                .unwrap_or_else(|| panic!("No previous (!= {next}) in {:?}", nbrs)) as usize;
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
