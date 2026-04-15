use pars3d::{FaceKind, parse_args};
use std::collections::HashMap;

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[Untriangulate Grid]: Untriangulate a grid which was triangulated by fusing sequential faces.",
      Input("-i", "--input"; "Input Image") => input : String = String::new(),
      Output("-o", "--output"; "Output Mesh") => output : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }
    println!("[INFO]: Untriangulating {} -> {}", args.input, args.output);

    let mut input_scene = pars3d::load(&args.input).unwrap();
    let mut input_mesh = input_scene.meshes.pop().unwrap();

    let ff_adj = input_mesh.face_face_adj();

    let mut pairs = HashMap::new();
    for fi in 0..input_mesh.f.len() {
        if pairs.contains_key(&fi) {
            continue;
        }
        let f = &input_mesh.f[fi];
        let mut any = false;
        for &ofi in ff_adj.adj(fi) {
            let ofi = ofi as usize;
            let o_f = &input_mesh.f[ofi];
            let Some(vis) = f.shared_edge(o_f) else {
                continue;
            };
            let [v0, v1] = vis.map(|vi| input_mesh.v[vi]);

            if v0[0] == v1[0] || v0[1] == v1[1] {
                continue;
            }

            assert_eq!(pairs.insert(fi, ofi), None);
            assert_eq!(pairs.insert(ofi, fi), None);
            any = true;
            break;
        }
        assert!(any);
    }
    let mut new_f = vec![];
    for (fi0, fi1) in pairs.into_iter() {
        if fi0 > fi1 {
            continue;
        }
        // TODO make this more robust
        let [a, b, _c] = input_mesh.f[fi0].as_tri().unwrap();
        let [_d, e, f] = input_mesh.f[fi1].as_tri().unwrap();
        new_f.push(FaceKind::Quad([a, b, e, f]));
    }

    input_mesh.f = new_f;

    input_scene.meshes.push(input_mesh);
    input_scene.textures[0].original_path =
        std::path::Path::new(&input_scene.textures[0].original_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
    pars3d::save(args.output, &input_scene, false).unwrap();
    Ok(())
}
