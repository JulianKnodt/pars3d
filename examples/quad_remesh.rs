#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{load, normalize, save};
    use std::collections::BTreeSet;

    let scale = 0.01;
    macro_rules! help {
        () => {{
            help!("");
        }};
        ($err: expr) => {{
            if $err != "" {
                eprintln!("[ERROR]: {}", $err);
            }
            eprintln!("[]: Quad Remeshes <arg1> to <arg2>");
            eprintln!("Usage: <bin> src dst");
            return Ok(());
        }};
    }

    #[derive(PartialEq, Eq)]
    pub enum State {
        Empty,
    }

    let mut state = State::Empty;
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        match v.as_str() {
            "-h" | "--help" => help!(),
            v if v.starts_with('-') => help!(format!("Unknown flag {v}")),
            _ => {}
        }
        match state {
            State::Empty => {
                if src.is_none() {
                    src = Some(v);
                } else if dst.is_none() {
                    dst = Some(v)
                } else {
                    help!("Too many arguments");
                };
            }
        }
    }
    if state != State::Empty {
        help!();
    }
    let Some(src) = src else {
        help!();
    };
    let Some(dst) = dst else {
        help!();
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
    args.orientation_smoothing_iters = 0;
    let new_vertices = im::instant_mesh(&m.v, &m.f, &vn, &mut field, rand::random, &args);

    m.v = new_vertices;

    m.repopulate_scene(&mut scene);

    save(dst, &scene)
}
