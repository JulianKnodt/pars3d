use pars3d::fbx;

fn main() {
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst.is_none() {
            dst = Some(v)
        } else {
            eprintln!("Usage: <bin> src dst");
            return;
        };
    }
    let Some(src) = src else {
        eprintln!("Usage: <bin> src dst");
        return;
    };
    let Some(dst) = dst else {
        eprintln!("Usage: <bin> src dst");
        return;
    };
    println!("[INFO]: {src} -> {dst}");

    let scene = fbx::parser::load(&src).expect("Failed to load FBX scene");
    println!("# Meshes {:?}", scene.meshes.len());
    println!("# Nodes {:?}", scene.nodes.len());
    let out = std::fs::File::create(dst).expect("Failed to create file");
    fbx::export::export_fbx(&scene, std::io::BufWriter::new(out)).expect("Failed to save scene");
}
