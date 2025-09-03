#[cfg(not(feature = "svg"))]
fn main() {
    eprintln!("Not compiled with SVG support, not saving");
}

#[cfg(feature = "svg")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, load, parse_args, svg::save_uv};
    let args = parse_args!(
      "[INFO]: Save the UV of a mesh into an SVG",
      Input("-i", "--input") => input : String = String::new(),
      Output("-o", "--output") => output : String = String::new(),
      Width("-w", "--width") => width : F = 1.0,
      UseXY("--use-xy") => use_xy: bool = false,
    );

    if args.input.is_empty() || args.output.is_empty() {
        help!();
    }
    if !args.output.ends_with(".svg") {
        help!("[ERROR]: Output must be a .svg file");
    }
    println!("[INFO]: Rendering UV of {} to {}", args.input, args.output);

    let scene = load(&args.input).expect("Failed to load scene");
    let mesh = &scene.meshes[0];
    let width = args.width;

    if args.use_xy {
        let get_xy = |i: usize| std::array::from_fn(|d: usize| mesh.v[i][d]);
        save_uv(&args.output, get_xy, &mesh.f, width)
    } else {
        save_uv(&args.output, |i| mesh.uv[0][i], &mesh.f, width)
    }
}
