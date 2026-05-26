#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, FaceKind, add, load, parse_args, save};

    let args = parse_args!(
      "[INFO]: Make the texture of the input mesh constant per face",
      Input("-i", "--input"; "Input Mesh") => input : String = String::new(),
      Output("-o", "--output"; "Output path to wireframe mesh") => output : String = String::new(),
      Image("--image"; "Source image to use")
        => image : String = String::new(),
    );

    if args.input.is_empty() {
        help!("Missing an input mesh");
    }
    if args.output.is_empty() {
        help!("Missing output destination");
    }
    let src = args.input;
    let dst = args.output;
    let img = image::open(args.image)
        .expect("Failed to open --image")
        .to_rgb8();

    let scene = load(&src).expect("Failed to load input scene");
    let m = scene.into_flattened_mesh();

    let mut per_face_colors = vec![[0.; 3]; m.f.len()];
    let num_samples = 256;
    for (fi, f) in m.f.iter().enumerate() {
        // random samples (biased)
        for t in f.as_triangle_fan().map(FaceKind::Tri) {
            let t_uv = t.map_kind(|vi| m.uv[0][vi]);
            for _ in 0..num_samples {
                let u: F = rand::random();
                let v: F = rand::random();
                let u = u.sqrt();
                let b0 = 1. - u;
                let b1 = u * (1. - v);
                let b2 = u * v;

                let [u, v] = t_uv.from_barycentric_tri([b0, b1, b2]);

                let Some(image::Rgb(rgb)) = image::imageops::sample_bilinear(&img, u, v) else {
                    continue;
                };
                let rgb = rgb.map(|v| v as f32 / 255.);
                per_face_colors[fi] = add(per_face_colors[fi], rgb);
            }
        }
    }

    for (fi, f) in m.f.iter().enumerate() {
        let denom = (num_samples * f.num_tris()) as F;
        per_face_colors[fi] = per_face_colors[fi].map(|v| v / denom);
    }

    let out_mesh = m.with_face_coloring(&per_face_colors);

    save(dst, &out_mesh.into_scene(), true)
}
