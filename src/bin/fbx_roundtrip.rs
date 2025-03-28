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
        println!(
            "\t#V = {}, #F = {}, name = {}",
            m.v.len(),
            m.f.len(),
            m.name
        );
    }
    println!("# Nodes {:?}", scene.nodes.len());
    /*
    for n in &scene.nodes {
      println!("\tNode: ID = {:?}", n.id);
    }
    */

    println!("# Skins {:?}", scene.skins.len());
    println!("# Clusters {:?}", scene.clusters.len());
    println!("# Poses {:?}", scene.poses.len());
    println!("# AnimStacks {:?}", scene.anim_stacks.len());
    println!("# AnimLayers {:?}", scene.anim_layers.len());
    println!("# AnimCurves {:?}", scene.anim_curves.len());
    println!("# AnimCurveNodes {:?}", scene.anim_curve_nodes.len());
    println!("# Blendshapes {:?}", scene.blendshapes.len());
    println!(
        "# Blendshape Channels {:?}",
        scene.blendshape_channels.len()
    );
    let out = std::fs::File::create(dst).expect("Failed to create file");
    fbx::export::export_fbx(&scene, std::io::BufWriter::new(out)).expect("Failed to save scene");
}
