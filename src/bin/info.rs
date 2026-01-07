use pars3d::load;
use pars3d::parse_args;

use std::io::Write;

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "Utility for getting information on a mesh",
      Input("-i", "--input"; "Input Mesh") => input : String = String::new(),
      Dest("-s", "--stats"; "Stat file") => stats : String = String::new(),
    );

    if args.input.is_empty() {
        help!();
    };
    let src = args.input;
    println!("[INFO]: Info about {src}:");

    let scene = load(&src).expect("Failed to load input scene");
    println!("#Meshes = {}", scene.meshes.len());
    println!("#Nodes = {}", scene.nodes.len());
    println!("#V = {}", scene.num_vertices());
    println!("#E = {}", scene.num_edges());
    println!("#F = {}", scene.num_faces());

    let aabb = scene.aabb();
    println!("AABB = {aabb:?}");

    if scene.num_faces() == 0 {
        return Ok(());
    }
    println!("- Geometry Info:");
    let mut mesh = scene.clone().into_flattened_mesh();
    mesh.geometry_only();
    let mut sum_of_all_chord_lens = 0;
    let num_chords = mesh.num_quad_chords(|len| {
        sum_of_all_chord_lens += len;
    });
    println!("#Chords = {num_chords}");
    if num_chords != 0 {
        println!(
            "Avg Chord Len = {}",
            sum_of_all_chord_lens as f64 / num_chords as f64
        );
    }

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

    let dst_json = args.stats;
    if dst_json.is_empty() {
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
