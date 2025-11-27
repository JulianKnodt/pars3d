use pars3d::adjacency::vertex_vertex_adj;
use pars3d::geom_processing::{boundary_vertices, subdivision};
use pars3d::{F, Mesh, add, barycentric_2d, divk, kahan, parse_args};

use std::collections::{BTreeSet, VecDeque};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[subdiv]: Subdivide a mesh",
      Input("-i", "--input"; "Input mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output subdivided mesh") => output : String = String::new(),
      Eps("--eps"; "Epsilon for subdivision") => eps : F = 0.1,
      ClampTo("--clamp"; "Minimum Value to clamp output") => clamp : F = 2.,
      Triangulate("--tri"; "Triangulate input") => triangulate : bool = false => true,
      SaveHoneycomb("--save-honeycomb"; "Save Honeycomb")
        => save_honeycomb : bool = false => true,
      SaveAsUV("--save-uv"; "Save output as UV") => save_uv : bool = false => true,
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }

    let input = pars3d::load(args.input).expect("Failed to load input");
    let mut input = input.into_flattened_mesh();
    if args.triangulate {
        input.triangulate(0);
    }
    let (mut new_uv, new_f) = subdivision::honeycomb(&input.uv[0], &input.f, (), args.eps);

    let vv = vertex_vertex_adj(new_uv.len(), &new_f);
    let mut neg = (0..new_uv.len()).collect::<VecDeque<_>>();
    let mut in_neg = (0..new_uv.len()).collect::<BTreeSet<_>>();

    while let Some(vi) = neg.pop_front() {
        in_neg.remove(&vi);
        assert_eq!(neg.len(), in_neg.len());
        // For 2D len should be 3 for non-boundary and 2 for boundary
        let adjs = vv.adj(vi);
        if adjs.len() != 3 {
            continue;
        }
        let adjs = std::array::from_fn(|i| adjs[i] as usize);
        let uv_adjs = adjs.map(|uvi| new_uv[uvi]);
        let bary = barycentric_2d(new_uv[vi], uv_adjs);
        let lim = 0.332; // 0.329; // 0.332
        if bary.into_iter().all(|v| v > lim) {
            continue;
        }

        let new_pos = add(uv_adjs[0], add(uv_adjs[1], uv_adjs[2]));
        let new_pos = divk(new_pos, 3.);

        assert!(new_pos.iter().copied().all(F::is_finite));
        new_uv[vi] = new_pos;
        for adj in adjs {
            if in_neg.insert(adj) {
                neg.push_back(adj);
            }
        }
    }

    let vert_faces = &new_f[input.f.len()..];

    macro_rules! extract_vi {
      ($vi: expr) => {{
        let f = &vert_faces[$vi];
        if let Some(&vi) = f.as_slice().iter().find(|&&vi| vv.degree(vi) == 2) {
            return new_uv[vi];
        }
        // average of adjacent face centroids
        let sum = kahan(f.as_slice().into_iter().map(|&vi| new_uv[vi]));
        divk(sum, f.len() as F)
      }}
    }
    /*
    macro_rules! extract_fi {
      ($fi: expr) => {{
        let f = &input.f[$fi];
        f.map_kind(|vi| extract_vi!(vi))
      }}
    }
    */
    /*
    let mut any = true;
    let mut it = 0;
    while any {
        any = false;
        // refine each vertex by compressing it to zero-area
        for vf in vert_faces {
            for _it in 0..3 {
                let vf = vf.as_slice();
                let l = vf.len();
                for i in 0..l {
                    let c = vf[i];
                    let p = vf[(i + l - 1) % l];
                    let n = vf[(i + 1) % l];
                    assert_ne!(p, c);
                    assert_ne!(c, n);
                    assert_ne!(p, n);
                    new_uv[c] = divk(add(new_uv[n], new_uv[p]), 2.);
                }
            }
        }

        for vi in 0..new_uv.len() {
            let adjs = vv.adj(vi);
            if adjs.len() != 3 {
                continue;
            }
            let adjs = std::array::from_fn(|i| adjs[i] as usize);
            let uv_adjs = adjs.map(|uvi| new_uv[uvi]);
            let bary = barycentric_2d(new_uv[vi], uv_adjs);
            if bary.into_iter().all(|v| v > 0.01) {
                continue;
            }
            any = true;

            let new_pos = add(uv_adjs[0], add(uv_adjs[1], uv_adjs[2]));
            let new_pos = divk(new_pos, 3.);

            new_uv[vi] = new_pos;
            for adj in adjs {
                if in_neg.insert(adj) {
                    neg.push_back(adj);
                }
            }
        }
        it += 1;
        if it > 200 {
          break;
        }
    }
    */
    /*

    // while any face is inverted, make each vertex-vertex more compressed
    let mut flipped = (0..input.f.len()).collect::<VecDeque<_>>();
    while let Some(fi) = flipped.pop_front() {
      let nf = extract_fi!(fi);
      let area = nf.area();
      if area >= 0. {
        continue;
      }
      // otherwise update adjacent vertices to be tighter
    }
    */

    let new_verts = (0..vert_faces.len())
        .map(|vi| extract_vi!(vi))
        .collect::<Vec<_>>();

    if args.save_uv {
        input.v = new_verts.iter().copied().map(|[x, y]| [x, y, 0.]).collect();
    }
    input.uv[0] = new_verts;

    let mut colors = vec![[0., 0.3, 0.]; input.v.len()];
    let mut hc_colors = vec![[0., 0.3, 0.]; new_uv.len()];

    // check local injectivity
    //let bd_verts: BTreeSet<_> = boundary_vertices(&input.f).collect();
    let mut cnt = 0;
    for f in &input.f {
        let a = f.area_2d(&input.uv[0]);
        if a >= 0. {
            continue;
        }
        cnt += 1;

        for &vi in f.as_slice() {
            colors[vi] = [1., 0., 0.];
            for &hvi in new_f[input.f.len() + vi].as_slice() {
                hc_colors[hvi] = [1., 0., 0.];
            }
        }
    }
    if cnt > 0 {
      println!("Flip count: {cnt:?}");
    }
    input.vert_colors = colors;

    if args.save_honeycomb {
        let new_verts = new_uv.iter().map(|&[x, y]| [x, y, 0.]).collect();
        let mut m = Mesh::new_geometry(new_verts, new_f);
        m.uv[0] = new_uv;
        m.vert_colors = hc_colors;
        return pars3d::save(args.output, &m.into_scene());
    }

    pars3d::save(&args.output, &input.into_scene())
}
