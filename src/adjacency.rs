use super::{F, FaceKind, Mesh, dist, dot, normalize, sub};
use std::collections::{BTreeMap, BTreeSet};

/// Structure for maintaining adjacencies of a mesh with fixed topology.
/// If the mesh is modified, this structure will no longer be valid.
/// Can also be used to store associated data.
#[derive(Debug, Clone)]
pub struct Adj<D = ()> {
    /// Index and # of nbrs in adjacency vector
    idx_count: Vec<(u32, u16)>,
    /// Flattened 2D vector from vertex to its neighbors
    adj: Vec<u32>,
    /// Associated data with each edge
    data: Vec<D>,
}

pub fn vertex_vertex_adj(nv: usize, f: &[FaceKind]) -> Adj<()> {
    let mut nbrs = vec![vec![]; nv];
    for f in f {
        for [e0, e1] in f.edges() {
            assert_ne!(e0, e1);
            let e0_nbrs = &mut nbrs[e0];
            if !e0_nbrs.contains(&(e1 as u32)) {
                e0_nbrs.push(e1 as u32);
            }
            let e1_nbrs = &mut nbrs[e1];
            if !e1_nbrs.contains(&(e0 as u32)) {
                e1_nbrs.push(e0 as u32);
            }
        }
    }

    from_nbr_vec(&mut nbrs)
}

fn from_nbr_vec(nbrs: &mut Vec<Vec<u32>>) -> Adj<()> {
    let mut idx_count = vec![];
    let mut adj = vec![];
    for fi in 0..nbrs.len() {
        let Some(n) = nbrs.get_mut(fi) else {
            idx_count.push((0, 0));
            continue;
        };
        idx_count.push((adj.len() as u32, n.len() as u16));
        adj.append(n);
    }
    let data = vec![(); adj.len()];
    Adj {
        idx_count,
        adj,
        data,
    }
}

impl Mesh {
    /// Returns vertices adjacent to each vertex in the input mesh
    pub fn vertex_vertex_adj(&self) -> Adj<()> {
        vertex_vertex_adj(self.v.len(), &self.f)
    }
    /// Returns faces adjacent to each vertex in the input mesh
    pub fn vertex_face_adj(&self) -> Adj<()> {
        let mut nbrs = vec![vec![]; self.v.len()];
        for (fi, f) in self.f.iter().enumerate() {
            let fi = fi as u32;
            for e in f.edges() {
                for vi in e {
                    let vi_nbr = &mut nbrs[vi];
                    if !vi_nbr.contains(&fi) {
                        vi_nbr.push(fi);
                    }
                }
            }
        }

        from_nbr_vec(&mut nbrs)
    }
    pub fn face_face_adj(&self) -> Adj<()> {
        let edge_adjs = self.edge_kinds();
        let mut f_nbrs = vec![vec![]; self.f.len()];

        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                f_nbrs[fi].extend(
                    edge_adjs[&e]
                        .as_slice()
                        .iter()
                        .copied()
                        .filter(|&ofi| ofi != fi)
                        .map(|v| v as u32),
                );
            }
        }
        from_nbr_vec(&mut f_nbrs)
    }
    pub fn face_face_pos_adj(&self) -> Adj<()> {
        let edge_adjs = self.edge_pos_kinds();
        let mut f_nbrs = vec![vec![]; self.f.len()];

        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                let [e0, e1] = e.map(|vi| self.v[vi].map(F::to_bits));
                f_nbrs[fi].extend(
                    edge_adjs[&std::cmp::minmax(e0, e1)]
                        .as_slice()
                        .iter()
                        .copied()
                        .filter(|&ofi| ofi != fi)
                        .map(|v| v as u32),
                );
            }
        }
        from_nbr_vec(&mut f_nbrs)
    }
}

impl<D> Adj<D> {
    pub fn uniform(self) -> Adj<F>
    where
        D: Copy,
    {
        self.map(|_, _, _, _| 1.)
    }
    pub fn laplacian(self, f: &[FaceKind], v: &[[F; 3]]) -> Adj<F>
    where
        D: Copy,
    {
        const EPS: F = 2e-6;
        let mut per_edge_weights = BTreeMap::new();
        for f in f.iter() {
            for pvni @ [pi, _, ni] in f.incident_edges() {
                let [p, v, n] = pvni.map(|vi| v[vi]);
                let a = dist(p, v);
                let b = dist(v, n);
                let c = dist(p, n);
                let area = crate::herons_area([a, b, c]);
                let v = a * a + b * b - c * c;
                let cot_c = v / (4. * area + EPS);
                let ew = per_edge_weights
                    .entry(std::cmp::minmax(pi, ni))
                    .or_insert(0.);
                *ew += cot_c;
            }
        }

        let mut per_vert_weights = vec![0.; v.len()];
        for f in f.iter() {
            if f.len() == 0 {
                continue;
            }
            let mut total_area = 0.;
            for t in f.as_triangle_fan() {
                let [v0, v1, v2] = t.map(|vi| v[vi]);
                let es = [dist(v0, v1), dist(v1, v2), dist(v2, v0)];
                total_area += crate::herons_area(es);
            }
            total_area /= f.len() as F;
            for &vi in f.as_slice() {
                per_vert_weights[vi] += total_area;
            }
        }

        fn softplus(x: F) -> F {
            if x > 1. { x } else { (1. + x.exp()).ln() }
        }

        self.map(|_, vi0, vi1, _| {
            let voro = per_vert_weights[vi0];
            let l = per_edge_weights[&std::cmp::minmax(vi0, vi1)];
            softplus(l) / (2. * voro + EPS)
        })
    }

    pub fn len(&self) -> usize {
        self.idx_count.len()
    }
    /// Modifies the data for this vertex adjacency based on a function which takes an ordered
    /// edge. Allocates one vector for the new data.
    pub fn map<U>(self, f: impl Fn(&Self, usize, usize, D) -> U) -> Adj<U>
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

        Adj {
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

    /// Returns adjacent vertices to this vertex as a mutable slice.
    #[allow(unused)]
    pub(crate) fn adj_mut(&mut self, v: usize) -> &mut [u32] {
        let (idx, cnt) = self.idx_count[v];
        if cnt == 0 {
            return &mut [];
        }
        let idx = idx as usize;
        let cnt = cnt as usize;

        &mut self.adj[idx..idx + cnt]
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
    pub fn two_ring_faces<D2>(&self, vi: usize, vertex_face_adj: &Adj<D2>, buf: &mut Vec<usize>) {
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

        assert!(
            out.values()
                .all(|&[v0, v1]| v0 != usize::MAX && v1 != usize::MAX)
        );

        // Fix up the loop order
        let mut num_loops = 0;
        let mut not_visited = out.keys().copied().collect::<BTreeSet<_>>();
        while let Some(first) = not_visited.pop_first() {
            num_loops += 1;
            let mut prev: usize = first;
            let mut curr: usize = out[&first][1];
            while curr != first {
                not_visited.remove(&curr);
                if out[&curr][1] == prev {
                    out.get_mut(&curr).unwrap().swap(0, 1);
                }
                let next = out[&curr][1];
                prev = curr;
                curr = next;
            }
        }

        (num_loops, out)
    }

    /// For this adjacency, returns a labelling of each element's component, along
    /// with the total number of components present.
    pub fn connected_components(&self) -> (Vec<u32>, u32) {
        let mut unseen = (0..self.idx_count.len()).collect::<BTreeSet<_>>();

        let mut out = vec![u32::MAX; self.idx_count.len()];
        let mut curr = 0;
        while let Some(fst) = unseen.pop_first() {
            out[fst as usize] = curr;
            let mut stack = self.adj(fst).to_vec();
            while let Some(next) = stack.pop() {
                if !unseen.remove(&(next as usize)) {
                    assert_ne!(out[next as usize], u32::MAX);
                    continue;
                }
                assert_eq!(out[next as usize], u32::MAX);
                out[next as usize] = curr;
                stack.extend_from_slice(self.adj(next as usize));
            }
            curr += 1;
        }
        (out, curr)
    }
}

/// Given a tree which has all loops, returns a vertex in the longest loop of the set.
/// Distance is measured using some arbitrary function, which can map to euclidean distance in
/// 3D, a count of edges or some 2D length.
pub fn longest_loop(
    loops: &BTreeMap<usize, [usize; 2]>,
    dist_fn: impl Fn(usize, usize) -> F,
) -> Option<usize> {
    let mut seen = BTreeSet::new();
    let mut best = None;
    let mut best_dist = F::INFINITY;
    for &v in loops.keys() {
        if !seen.insert(v) {
            continue;
        }
        let mut prev = v;
        let mut n = loops[&v][0];
        let mut curr_dist = dist_fn(prev, n);
        while n != v {
            assert!(seen.insert(n));
            assert_ne!(prev, loops[&n][0]);
            prev = n;
            n = loops[&n][0];
            curr_dist += dist_fn(prev, n);
        }
        curr_dist += dist_fn(n, prev);
        if curr_dist < best_dist {
            best_dist = curr_dist;
            best = Some(v);
        }
    }
    best
}
