use super::aabb::AABB;
use super::edge::EdgeKind;
use super::{F, U, add, cross, dot, kmul, length, normalize, sub, tri_area, tri_area_2d};
use super::{FaceKind, Mesh, Scene, face::Barycentric};
use std::collections::BTreeMap;

/// Curvature related functions
pub mod curvature;

/// Subdivision
pub mod subdivision;

/// Construction of a KDTree for a mesh
#[cfg(feature = "kdtree")]
pub mod kdtree;

/// Remesh an input mesh with a vertex field into a new mesh.
pub mod instant_meshes;

/// A data structure which tracks collapsible edges.
pub mod collapsible;

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

/// For a given set of faces, computes an adjacency map between them.
pub fn edge_kinds(fs: &[FaceKind]) -> BTreeMap<[usize; 2], EdgeKind> {
    let mut edges: BTreeMap<[usize; 2], EdgeKind> = BTreeMap::new();
    for (fi, f) in fs.iter().enumerate() {
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

pub fn boundary_edges(fs: &[FaceKind]) -> impl Iterator<Item = [usize; 2]> + '_ {
    edge_kinds(fs)
        .into_iter()
        .filter_map(|(k, v)| v.is_boundary().then_some(k))
}

pub fn boundary_vertices(fs: &[FaceKind]) -> impl Iterator<Item = usize> + '_ {
    boundary_edges(fs).flat_map(|v| v.into_iter())
}

/// Deletes triangles which lay on a non-manifold edge and share all the same vertices with
/// another triangle.
pub fn delete_non_manifold_duplicates(fs: &mut Vec<FaceKind>) -> usize {
    let mut to_del = vec![];
    let ek = edge_kinds(fs);
    for (fi, f) in fs.iter().enumerate() {
        if to_del.contains(&fi) {
            continue;
        }
        let Some(e) = f.edges_ord().find(|e| ek[e].is_non_manifold()) else {
            continue;
        };

        let other = f
            .edges_ord()
            .filter(|&oe| oe != e)
            .find_map(|oe| ek[&oe].opposite(fi));
        let Some(other) = other else {
            continue;
        };
        assert_ne!(other, fi);
        let del = f.edges_ord().all(|e| ek[&e].as_slice().contains(&other));
        if !del {
            continue;
        }
        to_del.push(fi);
        to_del.push(other);
    }
    to_del.sort();
    let num_del = to_del.len();
    while let Some(d) = to_del.pop() {
        fs.swap_remove(d);
    }
    num_del
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
        let normal = f.normal(vs);
        if length(normal) == 0. {
            continue;
        }
        let normal = normalize(normal);
        let area = match kind {
            VertexNormalWeightingKind::Uniform => 1.,
            VertexNormalWeightingKind::Area => f.area(vs),
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

impl Mesh {
    /// Computes vertex normals into mesh.n for its faces and vertices
    pub fn vertex_normals(&mut self, kind: VertexNormalWeightingKind) -> bool {
        vertex_normals(&self.f, &self.v, &mut self.n, kind)
    }
    #[inline]
    pub fn triangulate_with_new_edges(&mut self, mut cb: impl FnMut([usize; 2])) {
        let nf = self.f.len();
        for i in 0..nf {
            if self.f[i].len() <= 3 {
                continue;
            }
            let mesh_idx = self.face_mesh_idx.get(i).copied();
            let mat_idx = self.mat_for_face(i);

            let f = &self.f[i];
            let s = f.as_slice();
            let first = s[0];
            for &v in &s[2..s.len() - 1] {
                cb([first, v])
            }

            let curr_nfs = self.f.len();

            let t0 = f.as_triangle_fan().map(FaceKind::Tri).next().unwrap();
            let f = std::mem::replace(&mut self.f[i], t0);
            self.f
                .extend(f.as_triangle_fan().map(FaceKind::Tri).skip(1));
            let new_nfs = self.f.len();

            self.face_mesh_idx
                .extend((curr_nfs..new_nfs).filter_map(|_| mesh_idx));

            if let Some(mi) = mat_idx {
                let last_range = self.face_mat_idx.last_mut().unwrap();
                if last_range.1 == mi {
                    last_range.0.end = new_nfs;
                } else {
                    self.face_mat_idx.push((curr_nfs..new_nfs, mi));
                }
            }
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

            let mesh_idx = self.face_mesh_idx.get(fi);
            let mat_idx = self.mat_for_face(fi);
            // special case quads
            let curr_nfs = self.f.len();
            if let FaceKind::Quad([f0, f1, f2, f3]) = self.f[fi] {
                use super::tri_normal;
                let [ti0, ti1] = [[f0, f1, f2], [f0, f2, f3]];
                let t01 = [ti0, ti1].map(|ti| ti.map(|vi| self.v[vi]));
                let [n0, n1] = t01.map(tri_normal).map(normalize);

                let [ti2, ti3] = [[f1, f2, f3], [f1, f3, f0]];
                let t23 = [ti2, ti3].map(|ti| ti.map(|vi| self.v[vi]));
                let [n2, n3] = t23.map(tri_normal).map(normalize);

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

                if let Some(&mesh_idx) = mesh_idx {
                    self.face_mesh_idx.push(mesh_idx);
                }
                if let Some(mi) = mat_idx {
                    let last_range = self.face_mat_idx.last_mut().unwrap();
                    if last_range.1 == mi {
                        last_range.0.end += 1;
                    } else {
                        self.face_mat_idx.push((curr_nfs..curr_nfs + 1, mi));
                    }
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
            let new_nfs = self.f.len();
            if let Some(&mesh_idx) = mesh_idx {
                self.face_mesh_idx
                    .extend((curr_nfs..new_nfs).map(|_| mesh_idx));
            }
            if let Some(mi) = mat_idx {
                let last_range = self.face_mat_idx.last_mut().unwrap();
                if last_range.1 == mi {
                    last_range.0.end = new_nfs;
                } else {
                    self.face_mat_idx.push((curr_nfs..new_nfs, mi));
                }
            }

            num_split += 1;
        }
        num_split
    }

    /// Deletes triangles which lay on a non-manifold edge and share all the same vertices with
    /// another triangle.
    pub fn delete_non_manifold_duplicates(&mut self) -> usize {
        delete_non_manifold_duplicates(&mut self.f)
    }

    /// Splits any UV polygon which is self intersecting
    /// Returning how many polygons were split.
    pub fn split_self_intersecting_uv_poly(&mut self, channel: usize) -> usize {
        let mut num_split = 0;
        let curr_f = self.f.len();
        for fi in 0..curr_f {
            if self.f[fi].is_tri() {
                continue;
            }

            let mesh_idx = self.face_mesh_idx.get(fi);
            let mat_idx = self.mat_for_face(fi);
            // special case quads
            let curr_nfs = self.f.len();
            if let FaceKind::Quad([f0, f1, f2, f3]) = self.f[fi] {
                let [ti0, ti1] = [[f0, f1, f2], [f0, f2, f3]];
                let t01 = [ti0, ti1].map(|ti| ti.map(|vi| self.uv[channel][vi]));
                let [n0, n1] = t01.map(tri_area_2d);

                let [ti2, ti3] = [[f1, f2, f3], [f1, f3, f0]];
                let t23 = [ti2, ti3].map(|ti| ti.map(|vi| self.uv[channel][vi]));
                let [n2, n3] = t23.map(tri_area_2d);
                if n0.signum() == n1.signum() && n2.signum() == n3.signum() {
                    continue;
                }

                let t01 = [ti0, ti1].map(|ti| ti.map(|vi| self.v[vi]));
                let t23 = [ti2, ti3].map(|ti| ti.map(|vi| self.v[vi]));
                let a01_min_area = t01.map(tri_area).into_iter().min_by(F::total_cmp).unwrap();
                let a23_min_area = t23.map(tri_area).into_iter().min_by(F::total_cmp).unwrap();

                if a01_min_area > a23_min_area {
                    self.f[fi] = FaceKind::Tri(ti0);
                    self.f.push(FaceKind::Tri(ti1));
                } else {
                    self.f[fi] = FaceKind::Tri(ti2);
                    self.f.push(FaceKind::Tri(ti3));
                }

                if let Some(&mesh_idx) = mesh_idx {
                    self.face_mesh_idx.push(mesh_idx);
                }
                if let Some(mi) = mat_idx {
                    let last_range = self.face_mat_idx.last_mut().unwrap();
                    if last_range.1 == mi {
                        last_range.0.end += 1;
                    } else {
                        self.face_mat_idx.push((curr_nfs..curr_nfs + 1, mi));
                    }
                }

                num_split += 1;
                continue;
            }

            let mut tri_fan = self.f[fi].as_triangle_fan();
            let Some(t0) = tri_fan.next() else {
                continue;
            };
            let area_signum =
                |t: [usize; 3]| tri_area_2d(t.map(|vi| self.uv[channel][vi])).signum();
            let s0 = area_signum(t0);
            let non_planar = tri_fan.any(|t| area_signum(t) != s0);
            if !non_planar {
                continue;
            }
            drop(tri_fan);
            let prev = std::mem::replace(&mut self.f[fi], FaceKind::empty());
            let mut tris = prev.as_triangle_fan();
            self.f[fi] = FaceKind::Tri(tris.next().unwrap());
            self.f.extend(tris.map(FaceKind::Tri));
            let new_nfs = self.f.len();
            if let Some(&mesh_idx) = mesh_idx {
                self.face_mesh_idx
                    .extend((curr_nfs..new_nfs).map(|_| mesh_idx));
            }
            if let Some(mi) = mat_idx {
                let last_range = self.face_mat_idx.last_mut().unwrap();
                if last_range.1 == mi {
                    last_range.0.end = new_nfs;
                } else {
                    self.face_mat_idx.push((curr_nfs..new_nfs, mi));
                }
            }

            num_split += 1;
        }
        num_split
    }

    /// Removes doublets (faces which share more than 1 edge)
    /// CAUTION: Allocates
    pub fn remove_doublets(&mut self) {
        let edge_adj = self.edge_kinds();

        // For each adjacent face, which edges are shared?
        let mut shared_edges: BTreeMap<usize, Vec<[usize; 2]>> = BTreeMap::new();

        let curr_f_len = self.f.len();
        for fi in 0..curr_f_len {
            shared_edges.clear();
            let f = &self.f[fi];
            for e in f.edges_ord() {
                for &adj_f in edge_adj[&e].as_slice() {
                    if adj_f == fi {
                        continue;
                    }
                    shared_edges.entry(adj_f).or_default().push(e);
                }
            }
            for (_, es) in shared_edges.iter() {
                // TODO should assume that more than two edges can be shared
                if es.len() != 2 {
                    assert!(es.len() < 2);
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
        use std::collections::HashMap;
        let mut counts = vec![];
        let mut idxs: HashMap<[usize; 2], u32> = HashMap::new();
        for f in &self.f {
            for [e0, e1] in f.edges_ord() {
                if e0 == e1 {
                    continue;
                }
                let idx = *idxs.entry([e0, e1]).or_insert_with(|| {
                    let idx = counts.len();
                    counts.push(0u8);
                    idx as u32
                }) as usize;
                let c = unsafe { counts.get_unchecked_mut(idx) };
                *c = c.saturating_add(1u8);
            }
        }
        let mut num_bd = 0;
        let mut num_manifold = 0;
        let mut num_nonmanifold = 0;
        for v in counts {
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
    /// Returns (#Manifold Edges, #Boundary Edges, #Nonmanifold Edges)
    pub fn num_edge_kinds_by_position(&self) -> (usize, usize, usize) {
        let mut edges: BTreeMap<[[U; 3]; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for e in f.edges_ord() {
                let [e0, e1] = e.map(|vi| self.v[vi].map(F::to_bits));
                let cnt = edges.entry(std::cmp::minmax(e0, e1)).or_default();
                *cnt += 1u32;
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
    pub fn num_boundary_edges(&self) -> usize {
        self.num_edge_kinds().1
    }

    /// Non-unique iterator over boundary edges
    pub fn boundary_edges(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        boundary_edges(&self.f)
    }

    /// Boundary vertices of this mesh
    pub fn boundary_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        boundary_vertices(&self.f)
    }

    /// Returns the associated face set with each edge.
    pub fn edge_kinds(&self) -> BTreeMap<[usize; 2], EdgeKind> {
        edge_kinds(&self.f)
    }

    /// Returns the associated face with each edge by each edge's position.
    pub fn edge_pos_kinds(&self) -> BTreeMap<[[U; 3]; 2], EdgeKind> {
        let mut edges: BTreeMap<[[U; 3]; 2], EdgeKind> = BTreeMap::new();
        for (fi, f) in self.f.iter().enumerate() {
            for e in f.edges_ord() {
                let [e0, e1] = e.map(|v| self.v[v].map(F::to_bits));
                edges
                    .entry(std::cmp::minmax(e0, e1))
                    .and_modify(|ek| {
                        ek.insert(fi);
                    })
                    .or_insert_with(|| EdgeKind::Boundary(fi));
            }
        }
        edges
    }

    /// Returns non-manifold edges and faces on those non-manifold edges
    pub fn non_manifold_faces(&self) -> impl Iterator<Item = ([usize; 2], Vec<usize>)> + '_ {
        self.edge_kinds().into_iter().filter_map(|(e, v)| {
            if let EdgeKind::NonManifold(fs) = v {
                Some((e, fs))
            } else {
                None
            }
        })
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
            let u = rng();
            let v = rng();

            let u = u.sqrt();
            let b0 = 1. - u;
            let b1 = u * (1. - v);
            let b2 = u * v;
            assert!((b0 + b1 + b2 - 1.).abs() < 1e-5);
            (*fi, Barycentric::new(&self.f[*fi], *ti, [b0, b1, b2]))
        })
    }

    pub fn aabb(&self) -> AABB<F, 3> {
        AABB::from_slice(&self.v)
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
        *v += agg;
        agg = *v;
    }
}
