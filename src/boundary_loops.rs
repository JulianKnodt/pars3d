
/*
    /// Computes all boundary loops for this mesh
    pub fn boundary_loops(&self) -> BoundaryLoops {
        let mut edge_counts: BTreeMap<[usize; 2], u32> = BTreeMap::new();
        for f in &self.f {
            for [e0, e1] in f.edges() {
                assert_ne!(e0, e1);
                let v = edge_counts.entry(std::cmp::minmax(e0, e1)).or_default();
                *v = *v + 1;
            }
        }

        let mut uniq_verts: HashSet<usize> = edge_counts
            .iter()
            .filter(|(_, v)| **v == 1)
            .flat_map(|(k, _)| k.into_iter())
            .copied()
            .collect();
        let mut loops = vec![];
        while !uniq_verts.is_empty() {
            let next = *uniq_verts.iter().next().unwrap();
            assert!(uniq_verts.remove(&next));
            let bloop = BTreeMap::new();
            loops.push(bloop);
            todo!();
        }

        BoundaryLoops { loops }
    }

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryLoops {
    loops: Vec<BTreeMap<usize, [usize; 2]>>,
}

impl BoundaryLoops {
    pub fn num_loops(&self) -> usize {
        self.loops.len()
    }
}
*/
