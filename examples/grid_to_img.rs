use image::{self, GenericImageView};
use pars3d::grid::grid_from_delta;
use pars3d::{F, FaceKind, Mesh, parse_args};

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
        [&input_pos_x, &input_pos_y].map(|input| f32::from_ne_bytes(input.get_pixel(i, j).0))
    });
    let faces = gf.into_iter().map(FaceKind::Quad).collect::<Vec<_>>();
    let tmp_v = gv.into_iter().map(|[u, v]| [u, v, 0.]).collect::<Vec<_>>();
    let mut m = Mesh::new_geometry(tmp_v, faces);
    let ic = &input_col;
    m.vert_colors = (0..h)
        .flat_map(move |j| {
            (0..w).map(move |i| {
                let [r, g, b, _] = ic.get_pixel(i, j).0;
                [r, g, b].map(|v| v as F / 255.)
            })
        })
        .collect();

    // have grid mesh, now convert it to to an image

    pars3d::save("tmp.obj", &m.into_scene()).expect("Failed to save output");
    Ok(())
}
