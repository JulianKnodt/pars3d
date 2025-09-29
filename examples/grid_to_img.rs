use image::{self, GenericImageView};
use pars3d::coloring::bake_vertex_colors_to_texture_exact;
use pars3d::grid::grid_from_delta;
use pars3d::{F, FaceKind, add, kmul, parse_args};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[deformed_to_img]: Convert deformed grid images to an image:",
      InputPosX("-ipx", "--input-pos-x"; "Input position image")
        => input_pos_x : String = String::new(),
      InputPosY("-ipy", "--input-pos-y"; "Input position image")
        => input_pos_y : String = String::new(),
      InputCol("-ic", "--input-col"; "Input position image") => input_col : String = String::new(),
      Width("-w", "--width"; "Output Width") => width : u32 = 1024,
      Height("-h", "--height"; "Output Height") => height : u32 = 1024,
      Output("-o", "--output"; "Output High-Res Image") => output : String = String::new(),
      Stats("--stats"; "Unused") => stats: String = String::new(),
    );

    if args.input_pos_x.is_empty()
        || args.input_pos_y.is_empty()
        || args.output.is_empty()
        || args.input_col.is_empty()
    {
        help!();
    }
    println!(
        "[INFO]: Inflating X:{}, Y:{}, C:{}, to {}",
        args.input_pos_x, args.input_pos_y, args.input_col, args.output
    );

    let input_pos_x = image::open(args.input_pos_x).expect("Failed to load input pos x");
    let input_pos_y = image::open(args.input_pos_y).expect("Failed to load input pos y");
    assert_eq!(input_pos_x.dimensions(), input_pos_y.dimensions());
    let (w, h) = input_pos_x.dimensions();

    let input_col = image::open(args.input_col).expect("Failed to load input color");

    let (gv, gf) = grid_from_delta(w, h, |[i, j]| {
        [&input_pos_x, &input_pos_y].map(|input| f32::from_ne_bytes(input.get_pixel(i, j).0) as F)
    });
    let gv_3d = gv
        .iter()
        .copied()
        .map(|[u, v]| [u, v, 0.])
        .collect::<Vec<_>>();
    let faces = gf.into_iter().map(FaceKind::Quad).collect::<Vec<_>>();
    let faces0 = faces
        .iter()
        .flat_map(|f| f.as_triangle_fan_from(0))
        .map(FaceKind::Tri)
        .collect::<Vec<_>>();
    let faces1 = faces
        .iter()
        .flat_map(|f| f.as_triangle_fan_from(1))
        .map(FaceKind::Tri)
        .collect::<Vec<_>>();

    let ic = &input_col;
    let vert_colors = (0..h)
        .flat_map(move |j| {
            (0..w).map(move |i| {
                let [r, g, b, _] = ic.get_pixel(i, j).0;
                [r, g, b].map(|v| v as F / 255.)
            })
        })
        .collect::<Vec<_>>();
    let [out_img0, out_img1] = [faces0, faces1].map(|f| {
        bake_vertex_colors_to_texture_exact(
            [args.width, args.height],
            &gv_3d,
            &gv,
            &f,
            &vert_colors,
        )
    });

    let mean_img = image::ImageBuffer::from_fn(args.width, args.height, |i, j| {
        let p0 = out_img0.get_pixel(i, j).0;
        let p1 = out_img1.get_pixel(i, j).0;
        let mean = kmul(0.5, add(p0, p1));
        image::Rgb(mean.map(|v| (v * 255.) as u8))
    });
    let mean_img = image::imageops::flip_vertical(&mean_img);

    let _ = mean_img.save(args.output);
    Ok(())
}
