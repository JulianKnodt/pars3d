#[cfg(not(feature = "svg"))]
fn main() {
    eprintln!("Not compiled with SVG support, not saving");
}

#[cfg(feature = "svg")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, load, parse_args, svg::save_uv};
    let args = parse_args!(
      "[render_uv]: Save the UV of a mesh into an SVG:",
      Input("-i", "--input"; "Input Mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output SVG") => output : String = String::new(),
      Width("-w", "--width"; "SVG rendered line width") => width : F = 1.0,
      UseXY("--use-xy"; "Use XY of mesh instead of UV") => use_xy: bool = false => true,
      Rescale("-r", "--rescale-n1_1-to-0_1"; "Rescale [-1,1] to [0,1]") => rescale: bool = false => true,
      Zoom("--zoom"; "Zoom by a specific scale around the center") => zoom: F = 1.,
      Stats("--stats"; "Unused") => stats: String = String::new(),
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

    macro_rules! rescale {
        ($arr: expr) => {{
            let arr = $arr.map(|v| v * args.zoom);
            if args.rescale {
                arr.map(|v| (v + 1.) / 2.)
            } else {
                arr
            }
        }};
    }

    if args.use_xy {
        let get_xy = |i: usize| rescale!(std::array::from_fn(|d: usize| mesh.v[i][d]));
        save_uv(&args.output, get_xy, &mesh.f, width)
    } else {
        save_uv(&args.output, |i| rescale!(mesh.uv[0][i]), &mesh.f, width)
    }
}
