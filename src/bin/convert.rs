use pars3d::{load, parse_args, save};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "Import and export a given mesh.",
      Input("-i", "--input"; "Input mesh"; |arg: &str| !arg.is_empty())
        => input : String = String::new(),
      Output("-o", "--output"; "Output mesh"; |arg: &str| !arg.is_empty())
        => output : String = String::new(),
      Triangulate("--triangulate"; "Triangulate the input mesh")
        => triangulate : bool = false => true,
    );
    let src = args.input;
    let dst = args.output;
    if src.starts_with("-") || dst.starts_with("-") {
        help!();
    }
    let triangulate = args.triangulate;
    println!("[INFO]: {src} -> {dst}");

    let mut scene = load(&src).expect("Failed to load scene");
    if triangulate {
        for m in &mut scene.meshes {
            m.triangulate(0);
        }
    }
    save(dst, &scene)
}
