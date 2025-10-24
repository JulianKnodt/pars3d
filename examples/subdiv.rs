use pars3d::geom_processing::{
    subdivision::{self, HoneycombCheck3},
    vertex_normals,
};
use pars3d::{F, Mesh, parse_args};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[subdiv]: Subdivide a mesh",
      Input("-i", "--input"; "Input mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output subdivided mesh") => output : String = String::new(),
      Eps("--eps"; "Epsilon for subdivision") => eps : F = 0.1,
      Triangulate("--tri"; "Triangulate input") => triangulate : bool = false => true,
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }

    let input = pars3d::load(args.input).expect("Failed to load input");
    let mut input = input.into_flattened_mesh();
    input.geometry_only();
    if args.triangulate {
        input.triangulate(0);
    }
    let mut vn = vec![];
    vertex_normals(&input.f, &input.v, &mut vn, Default::default());

    let hc = HoneycombCheck3 {
        vertex_normals: &vn,
    };

    let (new_v, new_f) = subdivision::honeycomb(&input.v, &input.f, hc, args.eps);

    let m = Mesh::new_geometry(new_v, new_f);

    pars3d::save(&args.output, &m.into_scene())
}
