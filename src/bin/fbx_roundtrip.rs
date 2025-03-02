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
    if matches!(src.as_str(), "-h" | "--h") {
        eprintln!("Usage: <bin> src dst");
        return;
    }
    let Some(dst) = dst else {
        eprintln!("Usage: <bin> src dst");
        return;
    };
    if matches!(dst.as_str(), "-h" | "--h") {
        eprintln!("Usage: <bin> src dst");
        return;
    }
    println!("[INFO]: {src} -> {dst}");

    let scene = fbx::parser::load(&src).expect("Failed to load FBX scene");
    println!("# Meshes {:?}", scene.meshes.len());
    for m in &scene.meshes {
      println!("\t#V = {}", m.v.len());
      println!("\t#F = {}", m.f.len());
    }
    println!("# Nodes {:?}", scene.nodes.len());
    for n in &scene.nodes {
      println!("\tMesh = {:?}", n.mesh);
    }
    let out = std::fs::File::create(dst).expect("Failed to create file");
    fbx::export::export_fbx(&scene, std::io::BufWriter::new(out)).expect("Failed to save scene");
}
