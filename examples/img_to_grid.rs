use image::{self, GenericImageView};
use pars3d::grid::new_grid;
use pars3d::mesh::{Material, Texture, TextureKind};
use pars3d::{F, FaceKind, Mesh, parse_args};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[Image to Grid]: Convert an image to a mesh:",
      Input("-i", "--input"; "Input Image")
        => input : String = String::new(),
      Width("--width"; "Output Width") => width : u32 = 1024,
      Height("--height"; "Output Height") => height : u32 = 1024,
      Output("-o", "--output"; "Output Mesh") => output : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
      Triangulate("--triangulate"; "Triangulate output") => tri : bool = false => true,
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }
    let input = image::open(&args.input).expect("Failed to load input color");
    let (iw, ih) = input.dimensions();
    let ar = iw as F / ih as F;

    let (gv, gf) = new_grid(args.width, args.height);
    let faces = gf.into_iter().map(FaceKind::Quad).collect::<Vec<_>>();
    let verts = gv
        .iter()
        .copied()
        .map(|[u, v]| [u * ar, v, 0.])
        .collect::<Vec<_>>();

    let nv = gv.len();
    let mut mesh = Mesh::new_geometry(verts, faces);
    mesh.uv[0] = gv;
    mesh.face_mat_idx = vec![(0..nv, 0)];

    if args.tri {
        mesh.triangulate(0);
    }

    let mut scene = mesh.into_scene();
    use std::path::Path;
    scene.textures.push(Texture::new(
        TextureKind::Diffuse,
        [1.; 4],
        Some(input),
        Path::new(&args.input)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string(),
    ));
    scene.materials.push(Material {
        textures: vec![0],
        name: String::from("image"),
        path: args.output.clone().replace(".obj", ".mtl"),
    });

    pars3d::save(args.output, &scene, false).unwrap();
    Ok(())
}
