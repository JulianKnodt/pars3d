#![feature(cmp_minmax)]
#![feature(btree_set_entry)]

use pars3d::geom_processing::{boundary_vertices, subdivision};
use pars3d::{F, Mesh, add, barycentric_2d, divk, kahan, kmul, parse_args, sub, length};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[subdiv]: Subdivide a mesh",
      Input("-i", "--input"; "Input mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output subdivided mesh") => output : String = String::new(),
      Eps("--eps"; "Epsilon for subdivision") => eps : F = 0.01,
      Triangulate("--tri"; "Triangulate input") => triangulate : bool = false => true,
      SaveInput("--save-input"; "Save Input") => save_input : String = String::new(),
      SaveInputHoneycomb("--save-input-honeycomb"; "Save Input Honeycomb") => save_input_honeycomb : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }

    let input = pars3d::load(args.input).expect("Failed to load input");
    let mut input = input.into_flattened_mesh();
    let in_bd = boundary_vertices(&input.f).collect::<BTreeSet<_>>();

    let mut in_cnt = 0;
    let mut in_bd_cnt = 0;
    let mut in_inv_faces = VecDeque::new();
    for (fi, f) in input.f.iter().enumerate() {
        if f.area_2d(&input.uv[0]) > 0. {
            continue;
        }
        in_inv_faces.push_back(fi);
        if f.as_slice().iter().any(|vi| in_bd.contains(vi)) {
            in_bd_cnt += 1;
        } else {
            in_cnt += 1;
        }
    }

    println!(
        "[INFO]: Input num verts = {}, num faces = {}",
        input.v.len(),
        input.f.len()
    );
    println!("[INFO]: Input internal num flipped = {in_cnt}");
    println!("[INFO]: Input boundary num flipped = {in_bd_cnt}");

    if args.triangulate {
        input.triangulate(0);
    }

    let (mut hc_v, hc_f) = subdivision::honeycomb(&input.uv[0], &input.f, (), args.eps);
    let vert_faces = &hc_f[input.f.len()..];

    let hc_adj = pars3d::adjacency::vertex_vertex_adj(hc_v.len(), &hc_f);
    //let hc_bd = boundary_vertices(&hc_f).collect::<BTreeSet<_>>();
    let hc_vi_to_in_vi = (0..input.v.len())
        .flat_map(|in_vi| {
            vert_faces[in_vi]
                .as_slice()
                .into_iter()
                .map(move |hc_vi| (hc_vi, in_vi))
        })
        .collect::<BTreeMap<_, _>>();

    if !args.save_input.is_empty() {
        println!("[INFO]: Saving post-processed input to {}", args.save_input);
        pars3d::save(&args.save_input, &input.clone().into_scene())?;
    }

    macro_rules! extract_vi {
        ($in_vi: expr) => {{
            let in_vi = $in_vi;
            if in_bd.contains(&in_vi) {
                input.uv[0][in_vi]
            } else {
                let vf = &vert_faces[in_vi].as_slice();
                divk(
                    kahan(vf.into_iter().copied().map(|vi| hc_v[vi])),
                    vf.len() as F,
                )
            }
        }};
    }

    // --- optimize

    // --- first fix boundary vertices
    for in_vi in 0..input.v.len() {
        if !in_bd.contains(&in_vi) {
            continue;
        }
        let mut any = true;
        while any {
            any = false;
            for &hc_vi in vert_faces[in_vi].as_slice() {
                hc_v[hc_vi] = input.uv[0][in_vi];
            }
        }
    }

    // -- save after boundary vertices are corrected
    if !args.save_input_honeycomb.is_empty() {
        println!("[INFO]: Saving output to {}", args.output);
        let new_verts = hc_v.iter().map(|&[x, y]| [x, y, 0.]).collect();
        let mut m = Mesh::new_geometry(new_verts, hc_f.clone());
        m.uv[0] = hc_v.clone();
        pars3d::save(&args.save_input_honeycomb, &m.into_scene())?;
    }

    // --- optimize internal vertices
    println!("Optimizing internal vertices");

    let mut inverted = VecDeque::new();
    let mut flipped = (0..input.v.len()).collect::<VecDeque<_>>();
    while let Some(in_vi) = flipped.pop_front() {
        if in_bd.contains(&in_vi) {
            // boundary vertices are fixed
            continue;
        }

        assert!(inverted.is_empty());
        let hc_vs = &vert_faces[in_vi];
        let all_verts = hc_vs.as_slice().into_iter().copied();
        let evens = all_verts.clone().enumerate().filter(|(i,_)| i % 2 == 0).map(|iv| iv.1);
        let odds = all_verts.clone().enumerate().filter(|(i,_)| i % 2 == 1).map(|iv| iv.1);
        inverted.extend(evens);
        inverted.extend(odds);
        while let Some(hc_vi) = inverted.pop_front() {
            let pos = hc_v[hc_vi];
            let adjs = hc_adj.adj(hc_vi);
            assert_eq!(adjs.len(), 3);
            let adjs = std::array::from_fn(|i| adjs[i] as usize);
            let adjs_pos = adjs.map(|a| hc_v[a]);
            let bary = barycentric_2d(pos, adjs_pos);
            if bary.into_iter().all(|v| v > 0.) {
                continue;
            }
            //hc_v[hc_vi] = divk(kahan(adjs_pos), 3.);
            let o = 0.05; //1e-2
            let total_ws = 2.0 + o;
            hc_v[hc_vi] = kahan(adjs.into_iter().map(|a| {
                let w = if hc_vi_to_in_vi[&a] == in_vi { 1. } else { o };
                kmul(w, hc_v[a])
            }));
            hc_v[hc_vi] = divk(hc_v[hc_vi], total_ws);
            for a in adjs.into_iter().filter(|a| hc_vi_to_in_vi[&a] == in_vi) {
                if !inverted.contains(&a) {
                    inverted.push_back(a);
                }
            }

            let &o_in = adjs
                .into_iter()
                .find_map(|a| hc_vi_to_in_vi.get(&a).filter(|&&a_vi| a_vi != in_vi))
                .unwrap();
            flipped.push_back(o_in);
        }

        let mut nearest = None;
        for pcni in hc_vs.incident_edges() {
            let area = pars3d::tri_area_2d(pcni.map(|i| hc_v[i]));
            // todo maybe any of the areas is flipped?
            if area > 0. {
                continue;
            }
            // TODO here need to detect 2 inversions point in opposite directions, and that may
            // indicate there is no valid position? i.e. kernel doesn't exist.
            let c = pcni[1];
            let &o_in = hc_adj
                .adj(c)
                .into_iter()
                .find_map(|&a| {
                    hc_vi_to_in_vi
                        .get(&(a as usize))
                        .filter(|&&a_vi| a_vi != in_vi)
                })
                .unwrap();
            let delta = sub(extract_vi!(o_in), hc_v[c]);
            if delta == [0.; 2] {
              continue;
            }
            nearest = match nearest {
                None => Some(delta),
                Some(prev) if length(delta) < length(prev) => Some(delta),
                Some(_) => continue,
            }
        }

        if let Some(delta) = nearest {
            if length(delta) < 1e-7 {
              break;
            }
            assert_ne!(delta, [0.; 2]);
            for &hc_vi in hc_vs.as_slice() {
                hc_v[hc_vi] = add(hc_v[hc_vi], kmul(1., delta));
            }
            flipped.push_back(in_vi);
        }
    }

    println!("[INFO]: Saving output to {}", args.output);
    let new_verts = hc_v.iter().map(|&[x, y]| [x, y, 0.]).collect();
    let mut m = Mesh::new_geometry(new_verts, hc_f);
    m.uv[0] = hc_v;
    return pars3d::save(args.output, &m.into_scene());
}
