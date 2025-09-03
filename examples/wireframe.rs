#![feature(cmp_minmax)]

use pars3d::{F, load, parse_args, save};
use std::collections::BTreeSet;

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[INFO]: Outputs the wireframe of <arg1> to <arg2>",
      Input("-i", "--input"; "Input Mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output path to wireframe mesh") => output : String = String::new(),
      Width("-w", "--width"; "Wireframe thickness") => width : F = 1.5e-3,
    );

    if args.input.is_empty() {
        help!("Missing an input mesh");
    }
    if args.output.is_empty() {
        help!("Missing output destination");
    }
    let src = args.input;
    let dst = args.output;

    let scene = load(&src).expect("Failed to load input scene");
    let m = scene.into_flattened_mesh();

    let edges =
        m.f.iter()
            .flat_map(|f| f.edges_ord())
            .collect::<BTreeSet<_>>();

    let wf = pars3d::visualization::colored_wireframe(
        edges.iter().copied(),
        |vi| m.v[vi],
        |[_, _]| [0.; 3],
        args.width,
    );

    let out_mesh = pars3d::visualization::wireframe_to_mesh(wf);
    let out_scene = out_mesh.into_scene();

    save(dst, &out_scene)
}
