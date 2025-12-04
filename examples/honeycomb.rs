use pars3d::geom_processing::{boundary_vertices, subdivision};
use pars3d::{F, Mesh, parse_args};

use std::collections::{BTreeSet, VecDeque};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[subdiv]: Subdivide a mesh",
      Input("-i", "--input"; "Input mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output subdivided mesh") => output : String = String::new(),
      Eps("--eps"; "Epsilon for subdivision") => eps : F = 0.01,
      Triangulate("--tri"; "Triangulate input") => triangulate : bool = false => true,
      SaveInput("--save-input"; "Save Input") => save_input : String = String::new(),
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
        in_inv_faces.push_back((fi, false));
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

    let (hc_v, hc_f) = subdivision::honeycomb(&input.uv[0], &input.f, (), args.eps);

    if !args.save_input.is_empty() {
        return pars3d::save(&args.save_input, &input.clone().into_scene());
    }

    let new_verts = hc_v.iter().map(|&[x, y]| [x, y, 0.]).collect();
    let mut m = Mesh::new_geometry(new_verts, hc_f);
    m.uv[0] = hc_v;
    return pars3d::save(args.output, &m.into_scene());
}
