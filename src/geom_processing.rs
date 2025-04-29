use super::{cross, dot, normalize, sub, F, U};
use super::{FaceKind, Mesh};
use std::collections::BTreeMap;

impl Mesh {
    #[inline]
    pub fn triangulate_with_new_edges(&mut self, mut cb: impl FnMut([usize; 2])) {
        let nf = self.f.len();
        // nifty little method which doesn't require an extra buffer
        let mut i = 0;
        while i < nf {
            if self.f[i].len() <= 3 {
                i += 1;
                continue;
            }
            let f = self.f.swap_remove(i);
            let s = f.as_slice();
            let first = s[0];
            for &v in &s[2..s.len() - 2] {
                cb([first, v])
            }
            self.f.extend(f.as_triangle_fan().map(FaceKind::Tri));
        }
    }
    /// Splits non-planar faces into tris. Returns number of faces split.
    pub fn split_non_planar_faces(&mut self, eps: F) -> usize {
        assert!(eps > 0.);
        let mut num_split = 0;
        let m1eps = 1. - eps;
        let curr_f = self.f.len();
        for fi in 0..curr_f {
            if self.f[fi].is_tri() {
                continue;
            }
            let mut tri_fan = self.f[fi].as_triangle_fan();
            let Some(t0) = tri_fan.next() else {
                continue;
            };
            let get_normal = |t: [usize; 3]| {
                let [v0, v1, v2] = t.map(|vi| self.v[vi]);
                normalize(cross(sub(v2, v0), sub(v1, v0)))
            };
            let n0 = get_normal(t0);
            let non_planar = tri_fan.any(|t| dot(get_normal(t), n0) < m1eps);
            if !non_planar {
                continue;
            }
            drop(tri_fan); // not sure why this is explicitly needed
            let prev = std::mem::replace(&mut self.f[fi], FaceKind::empty());
            let mut tris = prev.as_triangle_fan();
            self.f[fi] = FaceKind::Tri(tris.next().unwrap());
            self.f.extend(tris.map(FaceKind::Tri));
            num_split += 1;
        }
        num_split
    }
    /// Removes doublets (faces which share more than 1 edge)
    /// CAUTION: Allocates
    pub fn remove_doublets(&mut self) {
        // TODO maybe easier to describe as degree 2 non border vertex
        let mut edge_adj: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
        // face index -> Vec<usize> (adjacent faces)
        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                edge_adj.entry(e).or_default().push(fi);
            }
        }

        let mut shared_edges: BTreeMap<usize, Vec<[usize; 2]>> = BTreeMap::new();
        let curr_f_len = self.f.len();
        for fi in 0..curr_f_len {
            shared_edges.clear();
            let f = &self.f[fi];
            for e in f.edges_ord() {
                for &adj_f in &edge_adj[&e] {
                    if adj_f == fi {
                        continue;
                    }
                    shared_edges.entry(adj_f).or_default().push(e);
                }
            }
            for (_, es) in shared_edges.iter() {
                if es.len() != 2 {
                    continue;
                }
                let [e0, e1] = [es[0], es[1]];
                let shared = match [e0, e1] {
                    [[a, _] | [_, a], [b, _] | [_, b]] if a == b => a,
                    _ => continue,
                };
                let f = std::mem::replace(&mut self.f[fi], FaceKind::empty());
                let mut edges = f.edges().filter(|e| !e.contains(&shared));
                let [v0, v1] = edges.next().unwrap();
                // Replace first face directly.
                self.f[fi] = FaceKind::Tri([v0, v1, shared]);
                for [v0, v1] in edges {
                    self.f.push(FaceKind::Tri([v0, v1, shared]));
                }
                break;
            }
        }
    }

    /// Reports the number of doublets in this mesh.
    /// CAUTION: Allocates
    pub fn num_doublets(&self) -> usize {
        let mut edge_face_count: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        let mut vert_count = vec![0; self.v.len()];
        for f in &self.f {
            for e in f.edges_ord() {
                *edge_face_count.entry(e).or_default() += 1;
                for vi in e {
                    vert_count[vi] += 1;
                }
            }
        }
        // ignore boundary faces
        for (e, c) in edge_face_count {
            if c == 1 {
                for vi in e {
                    vert_count[vi] = usize::MAX;
                }
            }
        }

        vert_count.into_iter().filter(|&cnt| cnt == 4).count()
    }

    /// Returns (#Boundary Edges, #Manifold Edges, #Nonmanifold Edges)
    pub fn num_edge_kinds(&self) -> (usize, usize, usize) {
        let mut edges: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for e in f.edges_ord() {
                let cnt = edges.entry(e).or_default();
                *cnt = *cnt + 1u32;
            }
        }
        let mut num_bd = 0;
        let mut num_manifold = 0;
        let mut num_nonmanifold = 0;
        for v in edges.values() {
            let cnt = match v {
                0 => continue,
                1 => &mut num_bd,
                2 => &mut num_manifold,
                _ => &mut num_nonmanifold,
            };
            *cnt += 1;
        }
        (num_bd, num_manifold, num_nonmanifold)
    }
    /// Returns (#Boundary Edges, #Manifold Edges, #Nonmanifold Edges)
    pub fn num_edge_kinds_by_position(&self) -> (usize, usize, usize) {
        let mut edges: BTreeMap<[[U; 3]; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for e in f.edges_ord() {
                let [e0, e1] = e.map(|vi| self.v[vi].map(F::to_bits));
                let cnt = edges.entry(std::cmp::minmax(e0, e1)).or_default();
                *cnt = *cnt + 1u32;
            }
        }
        let mut num_bd = 0;
        let mut num_manifold = 0;
        let mut num_nonmanifold = 0;
        for v in edges.values() {
            let cnt = match v {
                0 => continue,
                1 => &mut num_bd,
                2 => &mut num_manifold,
                _ => &mut num_nonmanifold,
            };
            *cnt += 1;
        }
        (num_bd, num_manifold, num_nonmanifold)
    }
    pub fn num_boundary_edges(&self) -> usize {
        self.num_edge_kinds().0
    }

    /// Non-unique iterator over boundary vertices
    pub fn boundary_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        let mut edges: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for [e0, e1] in f.edges() {
                let cnt = edges.entry(std::cmp::minmax(e0, e1)).or_default();
                *cnt = *cnt + 1u32;
            }
        }
        edges
            .into_iter()
            .filter(|(_, v)| *v == 1)
            .flat_map(|(k, _)| k.into_iter())
    }
    /// Non-unique iterator over boundary vertices
    pub fn non_manifold_faces(&self) -> impl Iterator<Item = ([usize; 2], Vec<usize>)> + '_ {
        let mut edges: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
        for (fi, f) in self.f.iter().enumerate() {
            for [e0, e1] in f.edges() {
                let fs = edges.entry(std::cmp::minmax(e0, e1)).or_default();
                fs.push(fi);
            }
        }
        edges.into_iter().filter(|(_, v)| v.len() > 2)
    }
}
