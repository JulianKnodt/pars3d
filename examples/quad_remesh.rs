#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, load, normalize, save};

    let mut scale = 0.01;
    macro_rules! help {
        ($( $err: expr )?) => {{
            $(eprintln!("[ERROR]: {}", $err);)*
            eprintln!("[]: Quad Remeshes <arg1> to <arg2>");
            eprintln!("Usage: <bin> src dst [--scale <float> = {scale}]");
            return Ok(());
        }};
    }

    #[derive(PartialEq, Eq, Debug)]
    pub enum State {
        Empty,
        Scale,
    }

    let mut state = State::Empty;
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        match v.as_str() {
            "-h" | "--help" => help!(),
            "--scale" => {
                state = State::Scale;
                continue;
            }
            v if v.starts_with('-') => help!(format!("Unknown flag {v}")),
            _ => {}
        }
        match state {
            State::Scale => {
                scale = match v.parse::<F>() {
                    Ok(s) => s,
                    Err(e) => help!(format!("Failed to parse scale ({v:?}) as float, err: {e}")),
                };
            }
            State::Empty => {
                if src.is_none() {
                    src = Some(v);
                } else if dst.is_none() {
                    dst = Some(v)
                } else {
                    help!("Too many arguments (got {v})");
                };
            }
        }
    }
    if state != State::Empty {
        help!(format!(
            "Passed a flag, but did not get parameter ({state:?})"
        ));
    }
    let Some(src) = src else {
        help!("Missing source & dest paths");
    };
    let Some(dst) = dst else {
        help!("Missing dest path");
    };

    let mut scene = load(&src).expect("Failed to load input scene");
    let mut m = scene.into_flattened_mesh();
    m.geometry_only();

    let mut vn = vec![];
    pars3d::geom_processing::vertex_normals(&m.f, &m.v, &mut vn, Default::default());

    let mut field = (0..m.v.len())
        .map(|_| normalize(std::array::from_fn(|_| rand::random())))
        .collect::<Vec<_>>();

    use pars3d::geom_processing::instant_meshes as im;
    let mut args = im::Args::default();
    args.scale = scale;
    args.orientation_smoothing_iters = 50;
    let new_vertices = im::instant_mesh(&m.v, &m.f, &vn, &mut field, rand::random, &args);

    m.v = new_vertices;

    m.repopulate_scene(&mut scene);

    save(dst, &scene)
}
