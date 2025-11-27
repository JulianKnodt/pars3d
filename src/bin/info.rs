use pars3d::load;
use std::io::Write;

fn main() -> std::io::Result<()> {
    macro_rules! help {
        () => {{
            eprintln!("Usage: <bin> src [OPTIONAL dst_json]");
            return Ok(());
        }};
    }
    let mut src = None;
    let mut dst_json = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst_json.is_none() {
            dst_json = Some(v)
        } else {
            help!();
        };
    }
    let Some(src) = src else {
        help!();
    };
    if src.starts_with("-")
        || dst_json
            .as_ref()
            .map(|v| v.starts_with("-"))
            .unwrap_or(false)
    {
        help!();
    }
    println!("[INFO]: Info about {src}:");

    let scene = load(&src).expect("Failed to load input scene");
    println!("#Meshes = {}", scene.meshes.len());
    println!("#Nodes = {}", scene.nodes.len());
    println!("#V = {}", scene.num_vertices());
    println!("#E = {}", scene.num_edges());
    println!("#F = {}", scene.num_faces());
    println!("- Geometry Info:");
    let mut mesh = scene.clone().into_flattened_mesh();
    mesh.geometry_only();
    let mut sum_of_all_chord_lens = 0;
    let num_chords = mesh.num_quad_chords(|len| {
        sum_of_all_chord_lens += len;
    });
    println!("#Chords = {num_chords}");
    println!(
        "Avg Chord Len = {}",
        sum_of_all_chord_lens as f64 / num_chords as f64
    );

    for m in &scene.meshes {
        let (_, bd_e, nm_e) = m.num_edge_kinds();
        println!("#Boundary Edges: {bd_e}\n#Non-Manifold Edges {nm_e}");
        if nm_e != 0 {
            let eks = m.edge_kinds();
            for (_, ek) in eks.into_iter() {
                if !ek.is_non_manifold() {
                    continue;
                }
                for &fi in ek.as_slice() {
                    print!("{:?}, ", m.f[fi]);
                }
                println!();
            }
        }
    }

    let Some(dst_json) = dst_json else {
        return Ok(());
    };
    if std::fs::exists(&dst_json)? {
        todo!();
    } else {
        let mut dst = std::fs::File::create(dst_json).expect("Failed to create output json");
        writeln!(dst, "{{")?;
        writeln!(dst, "\t\"num_meshes\": {}", scene.meshes.len())?;
        writeln!(dst, "\t\"num_nodes\": {}", scene.nodes.len())?;
        writeln!(dst, "}}")?;
    }
    Ok(())
}
