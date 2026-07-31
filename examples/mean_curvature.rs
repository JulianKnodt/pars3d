use pars3d::{F, geom_processing::curvature::mean_curvature, parse_args};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[Mean Curvature]: Compute mean curvature on each vertex of a mesh",
      Input("-i", "--input"; "Input Mesh"; |v: &str| !v.is_empty()) => input : String = String::new(),
      Output("-o", "--output"; "Output Mesh"; |v: &str| !v.is_empty()) => output : String = String::new(),
    );

    let input = pars3d::load(args.input)?;

    let mut mesh = input.into_flattened_mesh();

    let mut dst = vec![];
    mean_curvature(&mesh.f, &mesh.v, &mut dst);

    let min = dst
        .iter()
        .map(|&v| pars3d::length(v))
        .min_by(|a, b| a.total_cmp(&b))
        .unwrap();
    let max = dst
        .iter()
        .map(|&v| pars3d::length(v))
        .max_by(|a, b| a.total_cmp(&b))
        .unwrap();

    let range = max - min;

    mesh.vert_colors.clear();
    mesh.vert_colors.extend(
        dst.into_iter()
            .map(|c| pars3d::coloring::magma((pars3d::length(c) - min) / range)),
    );

    pars3d::save(&args.output, &mesh.into_scene(), true)
}
