use super::aabb::AABB;
use super::edge::EdgeKind;
use super::{add, cross, dot, kmul, length, normalize, sub, tri_area, F, U};
use super::{face::Barycentric, FaceKind, Mesh, Scene};
use std::collections::BTreeMap;

/// Curvature related functions
pub mod curvature;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VertexNormalWeightingKind {
    Uniform,
    #[default]
    Area,
}

pub fn barycentric_areas(fs: &[FaceKind], vs: &[[F; 3]], dst: &mut Vec<F>) {
    dst.fill(0.);
    dst.resize(vs.len(), 0.);
    for f in fs {
        for t in f.as_triangle_fan() {
            let area = tri_area(t.map(|vi| vs[vi])) / 3.;
            for v in t {
                dst[v] += area;
            }
        }
    }
}

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
            for &v in &s[2..s.len() - 1] {
                cb([first, v])
            }
            self.f.extend(f.as_triangle_fan().map(FaceKind::Tri));
        }
    }

    /// Computes vertex normals into dst for a set of faces and vertices
    pub fn vertex_normals(
        fs: &[FaceKind],
        vs: &[[F; 3]],
        dst: &mut Vec<[F; 3]>,
        kind: VertexNormalWeightingKind,
    ) -> bool {
        dst.resize(vs.len(), [0.; 3]);
        dst.fill([0.; 3]);
        for f in fs {
            let normal = f.normal(&vs);
            if length(normal) < 1e-10 {
                continue;
            }
            let area = match kind {
                VertexNormalWeightingKind::Uniform => 1.,
                VertexNormalWeightingKind::Area => f.area(&vs),
            };
            for &vi in f.as_slice() {
                dst[vi] = add(dst[vi], kmul(area, normal));
            }
        }
        for n in dst {
            *n = normalize(*n);
        }
        true
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
            // special case quads
            if let FaceKind::Quad([f0, f1, f2, f3]) = self.f[fi] {
                use super::tri_normal;
                let [ti0, ti1] = [[f0, f1, f2], [f0, f2, f3]];
                let t01 = [ti0, ti1].map(|ti| ti.map(|vi| self.v[vi]));
                let [n0, n1] = t01.map(tri_normal).map(normalize);

                let [ti2, ti3] = [[f1, f2, f3], [f1, f3, f0]];
                let t23 = [ti2, ti3].map(|ti| ti.map(|vi| self.v[vi]));
                let [n2, n3] = t23.map(tri_normal);

                // actually probably only need one of these checks
                if dot(n0, n1) > m1eps && dot(n2, n3) > m1eps {
                    continue;
                }

                // split with largest min area
                let a01_min_area = t01.map(tri_area).into_iter().min_by(F::total_cmp).unwrap();
                let a23_min_area = t23.map(tri_area).into_iter().min_by(F::total_cmp).unwrap();

                if a01_min_area > a23_min_area {
                    self.f[fi] = FaceKind::Tri(ti0);
                    self.f.push(FaceKind::Tri(ti1));
                } else {
                    self.f[fi] = FaceKind::Tri(ti2);
                    self.f.push(FaceKind::Tri(ti3));
                }

                num_split += 1;
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

    /// Edge to adjacent faces
    pub(crate) fn edge_adj_map(&self) -> BTreeMap<[usize; 2], Vec<usize>> {
        let mut edge_adj: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                edge_adj.entry(e).or_default().push(fi);
            }
        }
        edge_adj
    }
    /// Removes doublets (faces which share more than 1 edge)
    /// CAUTION: Allocates
    pub fn remove_doublets(&mut self) {
        let edge_adj = self.edge_adj_map();

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

    /// Returns (#Manifold Edges, #Boundary Edges, #Nonmanifold Edges)
    pub fn num_edge_kinds(&self) -> (usize, usize, usize) {
        let mut edges: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for e in f.edges_ord() {
                if e[0] == e[1] {
                    continue;
                }
                let cnt = edges.entry(e).or_insert(0);
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
        (num_manifold, num_bd, num_nonmanifold)
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
        self.num_edge_kinds().1
    }

    /// Non-unique iterator over boundary vertices
    pub fn boundary_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        let mut edges: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for e in f.edges_ord() {
                *edges.entry(e).or_default() += 1;
            }
        }
        edges.into_iter().filter(|(_, v)| *v == 1).map(|(e, _)| e)
    }

    /// Returns the associated face set with each edge.
    pub fn edge_kinds(&self) -> BTreeMap<[usize; 2], EdgeKind> {
        let mut edges: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                edges
                    .entry(e)
                    .and_modify(|ek| {
                        ek.insert(fi);
                    })
                    .or_insert_with(|| EdgeKind::Boundary(fi));
            }
        }
        edges
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

    /// Given a lower bound on how many points should be returned, samples from each face based
    /// on its area. Must provide a sufficient RNG function which returns values in 0..1.
    pub fn random_points_on_mesh<'a>(
        &'a self,
        n: usize,
        mut rng: impl FnMut() -> F + 'a,
    ) -> impl Iterator<Item = (usize, Barycentric)> + 'a {
        assert!(
            !self.f.is_empty(),
            "Cannot sample random points without faces"
        );
        let mut areas = self
            .f
            .iter()
            .enumerate()
            .flat_map(|(fi, f)| {
                f.as_triangle_fan().enumerate().map(move |(ti, t)| {
                    let ta = tri_area(t.map(|vi| self.v[vi]));
                    (ta, fi, ti)
                })
            })
            .collect::<Vec<_>>();
        cumulative_sum(areas.iter_mut().map(|a| &mut a.0));
        assert!(areas.is_sorted());
        let max_area = areas.last().unwrap().0;
        for (a, _, _) in areas.iter_mut() {
            *a /= max_area;
        }

        (0..n).map(move |_| {
            let p = rng();
            assert!((0.0..=1.0).contains(&p));
            let idx = match areas.binary_search_by(move |a_rest| a_rest.0.partial_cmp(&p).unwrap())
            {
                Ok(i) => i,
                Err(e) => e.min(self.f.len() - 1),
            };
            let (_, fi, ti) = &areas[idx];
            let b0 = rng();
            let b1 = rng();
            let [b0, b1] = if b0 + b1 > 1. {
                [1. - b0, 1. - b1]
            } else {
                [b0, b1]
            };
            let b2 = (1. - b0 - b1).max(0.);
            (*fi, Barycentric::new(&self.f[*fi], *ti, [b0, b1, b2]))
        })
    }

    pub fn aabb(&self) -> AABB<F, 3> {
        let mut aabb = AABB::new();
        for &v in &self.v {
            aabb.add_point(v);
        }
        aabb
    }

    /// Implementation from: https://mobile.rodolphe-vaillant.fr/entry/20/compute-harmonic-weights-on-a-triangular-mesh
    /// Concept from:
    /// https://doc.cgal.org/latest/Weights/group__PkgWeightsRefMixedVoronoiRegionWeights.html
    pub fn mixed_voronoi_cell_area(&self) -> Vec<F> {
        let mut out = vec![0.; self.v.len()];
        for f in &self.f {
            for t in f.as_triangle_fan() {
                let tv = t.map(|vi| self.v[vi]);
                let angles: [_; 3] = std::array::from_fn(|i| {
                    let e0 = normalize(sub(tv[(i + 1) % 3], tv[i]));
                    let e1 = normalize(sub(tv[(i + 2) % 3], tv[i]));
                    dot(e0, e1).clamp(-1., 1.).acos()
                });

                let ta = tri_area(tv);
                const RIGHT: F = std::f64::consts::FRAC_PI_2 as F;
                if angles.iter().all(|&v| v < RIGHT) {
                    for vi in t {
                        out[vi] += ta / 3.;
                    }
                    continue;
                }

                for i in 0..3 {
                    out[t[i]] = ta / if angles[i] >= RIGHT { 2. } else { 4. }
                }
            }
        }
        out
    }
}

impl Scene {
    pub fn aabb(&self) -> AABB<F, 3> {
        self.meshes
            .iter()
            .map(Mesh::aabb)
            .fold(AABB::new(), |a, n| a.union(&n))
    }
}

fn cumulative_sum<'a>(vs: impl Iterator<Item = &'a mut F>) {
    let mut agg = 0.;
    for v in vs {
        *v = *v + agg;
        agg = *v;
    }
}
