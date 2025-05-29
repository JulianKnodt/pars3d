use super::{dot, normalize, sub, FaceKind, Mesh, F};
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

impl VertexAdj {
    pub fn new(fs: &[FaceKind], nv: usize) -> Self {
        let mut nbrs: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for f in fs {
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
        for vi in 0..nv {
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

impl Mesh {
    /// Returns vertices adjacent to each vertex in the input mesh
    pub fn vertex_vertex_adj(&self) -> VertexAdj<()> {
        VertexAdj::new(&self.f, self.v.len())
    }
    /// Returns faces adjacent to each vertex in the input mesh
    pub fn vertex_face_adj(&self) -> VertexAdj<()> {
        let mut nbrs: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for (fi, f) in self.f.iter().enumerate() {
            let fi = fi as u32;
            for e in f.edges() {
                for vi in e {
                    let vi_nbr = nbrs.entry(vi).or_default();
                    if !vi_nbr.contains(&fi) {
                        vi_nbr.push(fi);
                    }
                }
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

    /// Returns the unique set of vertices in the 2-ring around vi for a vertex-vertex adj.
    /// It's the caller's responsibility to clear out, if they so choose.
    #[inline]
    pub fn two_ring(&self, vi: usize, out: &mut Vec<usize>) {
        let prev = self.adj(vi);
        let curr_len = out.len();
        for &p in prev {
            for &a2_vi in self.adj(p as usize) {
                if prev.contains(&a2_vi) {
                    continue;
                }
                let a2_vi = a2_vi as usize;
                if a2_vi == vi {
                    continue;
                }
                if out[curr_len..].contains(&a2_vi) {
                    continue;
                }
                out.push(a2_vi);
            }
        }
    }

    /// Returns the unique set of faces which are adjacent to the one ring, but not adjacent to
    /// vi. It's the caller's responsibility to clear `buf`, if they so choose.
    pub fn two_ring_faces<D2>(
        &self,
        vi: usize,
        vertex_face_adj: &VertexAdj<D2>,
        buf: &mut Vec<usize>,
    ) {
        let face_adj = vertex_face_adj.adj(vi);
        let vert_adj = self.adj(vi);
        let start_len = buf.len();
        for &adj_v in vert_adj {
            let face_aa = vertex_face_adj.adj(adj_v as usize);
            for &a_f in face_aa {
                if face_adj.contains(&a_f) || buf[start_len..].contains(&(a_f as usize)) {
                    continue;
                }
                buf.push(a_f as usize);
            }
        }
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
    pub fn all_adj_data(&self) -> impl Iterator<Item = (usize, &[u32], &[D])> + '_ {
        (0..self.idx_count.len()).map(|vi| (vi, self.adj(vi), self.data(vi)))
    }

    /// Given an incoming edge to this vertex, return the edge that is closest to being
    /// opposite. Returns none if vertex does not share any faces
    pub fn regular_quad_vertex_opposing(
        &self,
        vert_face: &Self,
        vi: usize,
        prev: usize,
    ) -> Option<usize> {
        if self.degree(vi) != 4 {
            return None;
        }
        for v_adj in self.adj(vi) {
            let v_adj = *v_adj as usize;
            if v_adj == prev {
                continue;
            }
            let shares_face = vert_face
                .adj(v_adj)
                .iter()
                .any(|vif| vert_face.adj(prev).iter().any(|ovif| vif == ovif));
            if shares_face {
                continue;
            }
            return Some(v_adj);
        }
        None
    }
    /// Returns the approximate opposing vertex for a given previous vertex to
    /// vi. That is, which vertex is approximately going in the other direction
    /// (closest to 180 degrees). Returns the cos theta between the two of them
    pub fn approximate_opposing_vertex(&self, vs: &[[F; 3]], vi: usize, prev: usize) -> (usize, F) {
        let e_dir = normalize(sub(vs[vi], vs[prev]));
        let mut best = F::NEG_INFINITY;
        let mut best_vi = 0;
        for &adj in self.adj(vi) {
            let adj = adj as usize;
            let n_dir = normalize(sub(vs[adj], vs[vi]));
            let cos_theta = dot(e_dir, n_dir).clamp(-1., 1.);
            if cos_theta > best {
                best_vi = adj;
                best = cos_theta;
            }
        }
        assert_ne!(best, F::NEG_INFINITY);
        return (best_vi, best);
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
        let mut out: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
        for [e0, e1] in m.boundary_edges() {
            let slot = out
                .entry(e0)
                .or_insert([usize::MAX; 2])
                .iter_mut()
                .find(|v| **v == usize::MAX)
                .expect("Multiple boundary edges meet");
            *slot = e1;
            let slot = out
                .entry(e1)
                .or_insert([usize::MAX; 2])
                .iter_mut()
                .find(|v| **v == usize::MAX)
                .expect("Multiple boundary edges meet");
            *slot = e0;
        }

        assert!(out
            .values()
            .all(|&[v0, v1]| v0 != usize::MAX && v1 != usize::MAX));

        // Fix up the loop order
        let mut num_loops = 0;
        let mut not_visited = out.keys().copied().collect::<BTreeSet<_>>();
        while let Some(first) = not_visited.pop_first() {
            num_loops += 1;
            let mut prev: usize = first;
            let mut curr: usize = out[&first][1];
            while curr != first {
                if out[&curr][1] == prev {
                    out.get_mut(&curr).unwrap().swap(0, 1);
                }
                prev = curr;
                curr = out[&curr][1];
            }
        }

        (num_loops, out)
    }
}
