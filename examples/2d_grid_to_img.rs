#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use image;
    use pars3d::{F, FaceKind, add, divk, parse_args};

    use std::collections::BTreeSet;
    use std::collections::HashMap;
    let args = parse_args!(
      "[2D Grid to image]: Convert a grid into an image",
      Input("-i", "--input"; "Input Image")
        => input : String = String::new(),
      Output("-o", "--output"; "Output Mesh") => output : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.output.is_empty() || args.input.is_empty() {
        help!();
    }
    let input_scene = pars3d::load(&args.input).expect("Failed to load input scene");
    assert_eq!(input_scene.meshes.len(), 1);
    let m = &input_scene.meshes[0];

    let tex = input_scene.textures[0].image.as_ref().unwrap();

    let mut x_coords = BTreeSet::new();
    let mut y_coords = BTreeSet::new();
    let mut xy_to_face = HashMap::new();

    for (fi, f) in m.f.iter().enumerate() {
        let aabb = f.aabb(&m.v);
        let [x, y, _] = aabb.min;
        let x = (x * 500.).round() as u32;
        let y = (y * 500.).round() as u32;
        x_coords.insert(x);
        y_coords.insert(y);
        assert_eq!(None, xy_to_face.insert([x, y], fi));
    }

    let x_coords = x_coords.into_iter().collect::<Vec<_>>();
    let y_coords = y_coords.into_iter().collect::<Vec<_>>();

    let w = x_coords.len() as u32;
    let h = y_coords.len() as u32;
    assert_eq!(w * h, m.f.len() as u32);

    let mut per_fi_color = vec![[0.; 3]; m.f.len()];
    for (fi, f) in m.f.iter().enumerate() {
        let uv_f = f.map_kind(|vi| m.uv[0][vi]);
        for _ in 0..1024 {
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
            let image::Rgba(p) = image::imageops::sample_bilinear(tex, u, 1. - v).unwrap();
            let p = [p[0], p[1], p[2]].map(|v| v as F / 255.);
            per_fi_color[fi] = add(per_fi_color[fi], p);
        }
        per_fi_color[fi] = divk(per_fi_color[fi], 1024.);
    }

    let out_img = image::RgbImage::from_fn(w, h, |x, y| {
        let x = x_coords[x as usize];
        let y = y_coords[y as usize];
        let fi = xy_to_face[&[x, y]];
        image::Rgb(per_fi_color[fi].map(|v| (v.clamp(0., 1.) * 255.) as u8))
    });
    let out_img = image::imageops::flip_vertical(&out_img);
    out_img.save(&args.output).expect("Failed to save output");

    Ok(())
}
