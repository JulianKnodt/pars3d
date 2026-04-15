#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, FaceKind, Mesh, add, divk, parse_args};
    use std::collections::BTreeMap;
    let args = parse_args!(
      "[Bake Tex]: Bake textures from a mesh with the same # of faces.",

      Input("-i", "--input"; "Input Image") => input : String = String::new(),
      BF("--bake-from"; "Bake from this mesh (same # faces)")
        => bake_from : String = String::new(),
      Output("-o", "--output"; "Output Mesh") => output : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
      W("--width"; "Width of output image") => width : u32 = 1024,
      H("--height"; "Height of output image") => height : u32 = 1024,
      Subdivide("--subdiv"; "Subdivide the target this many times") => subdiv : u32 = 0,

      // TODO make this per area?
      SamplesPerFace("-spf"; "Samples per face") => samples_per_face: u32 = 128,
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }

    println!(
        "Baking from {} to {} -> {}",
        args.bake_from, args.input, args.output
    );

    let mut input_scene = pars3d::load(&args.input).unwrap();
    assert_eq!(input_scene.meshes.len(), 1);
    let input_mesh = &input_scene.meshes[0];

    let bake_scene = pars3d::load(&args.bake_from).unwrap();
    assert_eq!(bake_scene.meshes.len(), 1);
    let bake_mesh = &bake_scene.meshes[0];
    assert_eq!(input_mesh.f.len(), bake_mesh.f.len());
    assert_eq!(input_mesh.v.len(), bake_mesh.v.len());
    let in_tex = bake_scene.textures[0]
        .image
        .as_ref()
        .unwrap()
        .clone()
        .into_rgb32f();

    let mut key_to_fi = BTreeMap::new();
    let key = |f: &FaceKind, m: &Mesh| {
        let aabb = f.aabb(&m.v);
        let [lx, ly, lz] = aabb.min;
        let [hx, hy, hz] = aabb.max;
        [lx, ly, lz, hx, hy, hz].map(|v| (v * 1e3).floor() as u32)
    };

    for (fi, f) in bake_mesh.f.iter().enumerate() {
        let key = key(f, bake_mesh);
        assert_eq!(None, key_to_fi.insert(key, [fi, usize::MAX]));
    }
    for (fi, f) in input_mesh.f.iter().enumerate() {
        let key = key(f, input_mesh);
        let prev = key_to_fi.get_mut(&key).expect(&format!("{fi}"));
        assert_eq!(prev[1], usize::MAX);
        prev[1] = fi;
    }

    let mut per_face_color = vec![[0.; 3]; input_mesh.f.len()];
    // TODO handle quads here
    // bake one color to each triangle
    for fi in 0..bake_mesh.f.len() {
        let in_f = &bake_mesh.f[fi];

        let uv_f = in_f.map_kind(|vi| bake_mesh.uv[0][vi]);
        // Take a random triangle with 50
        for _ in 0..args.samples_per_face {
            let u: F = rand::random();
            let v: F = rand::random();
            let [u, v] = if u + v > 1. { [1. - u, 1. - v] } else { [u, v] };
            let w = 1. - u - v;
            let uv_t = FaceKind::Tri(
                uv_f.as_triangle_fan()
                    .nth(if rand::random() { 0 } else { 1 })
                    .unwrap(),
            );
            let [i, j] = uv_t.from_barycentric_tri([u, v, w]);
            let image::Rgb(rgb) = image::imageops::sample_bilinear(&in_tex, i, 1. - j).unwrap();
            per_face_color[fi] = add(rgb, per_face_color[fi]);
        }
    }

    for c in per_face_color.iter_mut() {
        *c = divk(*c, args.samples_per_face as F);
    }

    let mut out_img = image::ImageBuffer::<image::Rgb<F>, Vec<F>>::new(args.width, args.height);
    let mut out_cnt =
        image::ImageBuffer::<image::Luma<u32>, Vec<u32>>::new(args.width, args.height);

    for fi in 0..input_mesh.f.len() {
        let bake_f = &input_mesh.f[fi];
        let k = key(bake_f, &input_mesh);
        let in_fi = key_to_fi[&k][0];

        let face_color = per_face_color[in_fi];
        let uv_f = bake_f.map_kind(|vi| input_mesh.uv[0][vi]);
        for _ in 0..5000 {
            let u: F = rand::random();
            let v: F = rand::random();
            let [u, v] = if u + v > 1. { [1. - u, 1. - v] } else { [u, v] };
            let w = 1. - u - v;
            let uv_t = FaceKind::Tri(
                uv_f.as_triangle_fan()
                    .nth(if rand::random() { 0 } else { 1 })
                    .unwrap(),
            );
            let [u, v] = uv_t.from_barycentric_tri([u, v, w]);
            let u = (u * args.width as F).floor() as u32;
            let u = u.clamp(0, args.width - 1);
            let v = ((1. - v) * args.height as F).floor() as u32;
            let v = v.clamp(0, args.height - 1);
            *out_img.get_pixel_mut(u, v) = image::Rgb(add(out_img.get_pixel(u, v).0, face_color));
            out_cnt.get_pixel_mut(u, v).0[0] += 1;
        }
    }

    let mut out_u8 = image::RgbImage::new(args.width, args.height);
    for (x, y, p) in out_img.enumerate_pixels_mut() {
        let rgb = divk(p.0, out_cnt.get_pixel(x, y).0[0] as F);
        let rgb_u8 = rgb.map(|v| (v.clamp(0., 1.) * 255.) as u8);
        *out_u8.get_pixel_mut(x, y) = image::Rgb(rgb_u8);
    }

    input_scene.textures[0].image = Some(image::DynamicImage::ImageRgb8(out_u8));
    input_scene.textures[0].original_path = std::path::Path::new(&args.output)
        .with_extension("png")
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    input_scene.materials[0].path = args.output.clone().replace(".obj", ".mtl");
    pars3d::save(args.output, &input_scene, false).unwrap();
    println!("Done baking.");
    Ok(())
}
