#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use image::{GenericImage, GenericImageView};
    use pars3d::geom_processing::subdivision::{ident_subdiv, original_quad_index, quad_subdiv};
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
      Subdivide("--subdiv"; "Subdivide the target") => subdiv : bool = false => true,

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

    let (out_bs, out_fs) = if args.subdiv {
        let in_f = input_mesh
            .f
            .iter()
            .map(|f| {
                let &FaceKind::Quad(q) = f else {
                    todo!();
                };
                q
            })
            .collect::<Vec<_>>();
        quad_subdiv(&in_f)
    } else {
        let in_f = input_mesh
            .f
            .iter()
            .map(|f| {
                let &FaceKind::Quad(q) = f else {
                    todo!();
                };
                q
            })
            .collect::<Vec<_>>();
        (ident_subdiv(input_mesh.v.len()), in_f)
    };
    let out_vs = out_bs
        .iter()
        .map(|b| b.eval(&input_mesh.v))
        .collect::<Vec<_>>();
    let out_uvs = out_bs
        .iter()
        .map(|b| b.eval(&input_mesh.uv[0]))
        .collect::<Vec<_>>();

    let bake_scene = pars3d::load(&args.bake_from).unwrap();
    assert_eq!(bake_scene.meshes.len(), 1);
    let bake_mesh = &bake_scene.meshes[0];
    assert_eq!(input_mesh.f.len(), bake_mesh.f.len());
    assert_eq!(input_mesh.v.len(), bake_mesh.v.len());
    let bake_tex = bake_scene.textures[0]
        .image
        .as_ref()
        .unwrap()
        .clone()
        .into_rgb32f();

    /*
    let bw = bake_tex.width();
    let bh = bake_tex.height();
    */

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

    let mut out_img = image::ImageBuffer::<image::Rgb<F>, Vec<F>>::new(args.width, args.height);
    let mut out_cnt = image::ImageBuffer::<image::Luma<F>, Vec<F>>::new(args.width, args.height);

    const N: usize = 512;
    // for each output face
    for (o_fi, out_f) in out_fs.iter().enumerate() {
        let fi = original_quad_index(o_fi, if args.subdiv { 1 } else { 0 });

        let input_f = &input_mesh.f[fi];
        let k = key(input_f, &input_mesh);
        let bake_fi = key_to_fi[&k][0];
        assert_eq!(key_to_fi[&k][1], fi);
        let bake_f = &bake_mesh.f[bake_fi];

        let bake_uv_f = bake_f.map_kind(|vi| bake_mesh.uv[0][vi]);
        let bake_v_f = bake_f.map_kind(|vi| bake_mesh.v[vi]);

        let out_uv_f = FaceKind::Quad(out_f.map(|vi| out_uvs[vi]));
        let out_v_f = FaceKind::Quad(out_f.map(|vi| out_vs[vi]));
        //let out_f = FaceKind::Quad(*out_f);

        for _ in 0..N {
            let u: F = rand::random();
            let v: F = rand::random();
            let [u, v] = if u + v > 1. { [1. - u, 1. - v] } else { [u, v] };
            let w = 1. - u - v;

            let out_bary = [u, v, w];
            let tri = if rand::random() { 0 } else { 1 };
            let out_v_t = FaceKind::Tri(out_v_f.as_triangle_fan().nth(tri).unwrap());
            let pos = out_v_t.from_barycentric_tri(out_bary);

            let bake_bary = bake_v_f.barycentric(pos);
            let [bake_u, bake_v] = bake_uv_f.from_barycentric(bake_bary);
            let bake_u = bake_u.clamp(0., 1.);
            let bake_v = bake_v.clamp(0., 1.);

            let image::Rgb(src_pix) =
                image::imageops::sample_bilinear(&bake_tex, bake_u, 1. - bake_v).unwrap();

            let out_uv_t = FaceKind::Tri(out_uv_f.as_triangle_fan().nth(tri).unwrap());
            let [u, v] = out_uv_t.from_barycentric_tri(out_bary);
            let u = (u * args.width as F).clamp(0., (args.width - 1) as F);
            let v = ((1. - v) * args.height as F).clamp(0., (args.height - 1) as F);

            let u = u.round() as u32;
            let v = v.round() as u32;

            unsafe {
                let prev = out_img.unsafe_get_pixel(u, v).0;
                out_img.unsafe_put_pixel(u, v, image::Rgb(add(prev, src_pix)));
                let [prev_cnt] = out_cnt.unsafe_get_pixel(u, v).0;
                out_cnt.unsafe_put_pixel(u, v, image::Luma([prev_cnt + 1.]));
            }

            /*
            let uf = u.floor() as u32;
            let uc = (uf + 1).min(args.width - 1);
            let vf = v.floor() as u32;
            let vc = (vf + 1).min(args.height - 1);

            let ufw = u - uf as F;
            let vfw = v - vf as F;
            let ucw = (uf + 1) as F - u;
            let vcw = (vf + 1) as F - v;

            let wff = ucw * vcw;
            let wfc = ucw * vfw;
            let wcf = ufw * vcw;
            let wcc = ufw * vfw;

            let coord_and_weight = [
                ([uf, vf], wff),
                ([uf, vc], wfc),
                ([uc, vf], wcf),
                ([uc, vc], wcc),
            ];
            for ([u, v], w) in coord_and_weight {
                let col = pars3d::kmul(w, src_pix);
                unsafe {
                    let prev = out_img.unsafe_get_pixel(u, v).0;
                    out_img.unsafe_put_pixel(u, v, image::Rgb(add(prev, col)));
                    let [prev_cnt] = out_cnt.unsafe_get_pixel(u, v).0;
                    out_cnt.unsafe_put_pixel(u, v, image::Luma([prev_cnt + w]));
                }
            }
            */
        }
    }

    /*
    for (o_fi, out_f) in out_fs.iter().enumerate() {
        let out_uv_f = FaceKind::Quad(out_f.map(|vi| out_uvs[vi]));
        let face_color = out_face_colors[o_fi];
        for _ in 0..N {
            let u: F = rand::random();
            let v: F = rand::random();
            let [u, v] = if u + v > 1. { [1. - u, 1. - v] } else { [u, v] };
            let w = 1. - u - v;

            let out_bary = [u, v, w];
            let tri = if rand::random() { 0 } else { 1 };

            let out_uv_t = FaceKind::Tri(out_uv_f.as_triangle_fan().nth(tri).unwrap());
            let [u, v] = out_uv_t.from_barycentric_tri(out_bary);
            let u = (u * args.width as F).round().clamp(0., (args.width - 1) as F) as u32;
            let v = ((1. - v) * args.height as F).round().clamp(0., (args.height - 1) as F) as u32;

            *out_img.get_pixel_mut(u, v) = image::Rgb(add(out_img.get_pixel(u, v).0, face_color));
            out_cnt.get_pixel_mut(u, v).0[0] += 1;
        }
    }
    */

    let mut out_u8 = image::RgbImage::new(args.width, args.height);
    for (x, y, p) in out_img.enumerate_pixels_mut() {
        let cnt = out_cnt.get_pixel(x, y).0[0] as F;
        if cnt == 0. {
            continue;
        }
        let rgb = divk(p.0, cnt);
        let rgb_u8 = rgb.map(|v| (v.clamp(0., 1.) * 255.) as u8);
        *out_u8.get_pixel_mut(x, y) = image::Rgb(rgb_u8);
    }

    let n_out_f = out_fs.len();
    let out_fs = out_fs.into_iter().map(FaceKind::Quad).collect();
    input_scene.meshes[0] = Mesh::new_geometry(out_vs, out_fs);
    input_scene.meshes[0].uv[0] = out_uvs;
    input_scene.meshes[0].face_mat_idx = vec![(0..n_out_f, 0)];
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
