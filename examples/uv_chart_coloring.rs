#![feature(cmp_minmax)]

use pars3d::{F, U, load, save};
use std::collections::{BTreeMap, HashMap};

fn main() -> std::io::Result<()> {
    macro_rules! help {
        () => {{
            eprintln!("Outputs a mesh colored by which UV chart contains each face");
            eprintln!("Usage: <bin> src dst");
            return Ok(());
        }};
    }
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst.is_none() {
            dst = Some(v)
        } else {
            help!();
        };
    }
    let Some(src) = src else {
        help!();
    };
    let Some(dst) = dst else {
        help!();
    };

    let mut scene = load(&src).expect("Failed to load input scene");

    for m in &mut scene.meshes {
        m.clear_vertex_normals();
        let vv_adj = m.vertex_vertex_adj();
        let (comps, _) = vv_adj.connected_components();
        // TODO determine if two regions are adjacent (any two vertices with same position but
        // different chart)

        let mut ident_pos: HashMap<[U; 3], Vec<usize>> = HashMap::new();
        fn key(v: [F; 3]) -> [U; 3] {
            v.map(F::to_bits)
        }
        for vi in 0..m.v.len() {
            let k = key(m.v[vi]);
            ident_pos.entry(k).or_default().push(vi);
        }
        let mut adjs: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for vis in ident_pos.values() {
            for i in 0..vis.len() {
                let ci = comps[vis[i]] as usize;
                for j in i + 1..vis.len() {
                    let cj = comps[vis[j]] as usize;
                    if ci != cj {
                        adjs.entry(ci).or_default().push(cj);
                        adjs.entry(cj).or_default().push(ci);
                    }
                }
            }
        }

        // Apply greedy coloring, then set coloring per vertex.
        let coloring = pars3d::visualization::greedy_face_coloring(
            |vi| comps[vi] as usize,
            m.v.len(),
            |a| adjs.get(&a).map_or(&[], Vec::as_slice),
            &pars3d::coloring::HIGH_CONTRAST,
        );

        m.vert_colors = coloring;
    }

    save(dst, &scene).expect("Failed to save output");
    Ok(())
}
