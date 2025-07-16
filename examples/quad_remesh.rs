#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, load, normalize, save};
    use std::io::{BufRead, BufReader};

    let mut scale = 0.02;
    let mut field = None;
    macro_rules! help {
        ($( $err: tt )?) => {{
            $(eprintln!("[ERROR]: {}", format!($err));)*
            eprintln!("[]: Quad Remeshes <arg1> to <arg2>");
            eprintln!("Usage: <bin> src dst [--scale <float> = {scale}] [--field <CSV> = None]");
            return Ok(());
        }};
    }

    #[derive(PartialEq, Eq, Debug)]
    pub enum State {
        Empty,
        Scale,
        Field,
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
            "--field" => {
                state = State::Field;
                continue;
            }
            v if v.starts_with('-') => help!("Unknown flag {v}"),
            _ => {}
        }
        match state {
            State::Scale => {
                scale = match v.parse::<F>() {
                    Ok(s) => s,
                    Err(e) => help!("Failed to parse scale ({v:?}) as float, err: {e}"),
                };
            }
            State::Field => {
                field = Some(v);
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
        state = State::Empty;
    }
    if state != State::Empty {
        help!("Passed a flag, but did not get parameter ({state:?})");
    }
    let Some(src) = src else {
        help!("Missing source & dest paths");
    };
    let Some(dst) = dst else {
        help!("Missing dest path");
    };

    let scene = load(&src).expect("Failed to load input scene");
    let mut m = scene.into_flattened_mesh();
    m.geometry_only();
    m.triangulate();
    let (_s, _t) = m.normalize();

    let mut vn = vec![];
    pars3d::geom_processing::vertex_normals(&m.f, &m.v, &mut vn, Default::default());

    let mut field = if let Some(field) = field {
        let f = std::fs::File::open(field)?;
        let mut field = vec![];
        for l in BufReader::new(f).lines() {
            let l = l?;
            let mut keys = l.split(",");
            let mut vec = [0.; 3];
            for i in 0..3 {
                let Some(k) = keys.next() else {
                    eprintln!("Not enough values in --field <CSV> (expected 3 per line, got {i})");
                    return Ok(());
                };
                let Ok(k) = k.parse::<F>() else {
                    eprintln!("Expected numerical values in --field <CSV>, got {k}");
                    return Ok(());
                };
                vec[i] = k;
            }
            field.push(vec);
        }
        if field.len() != m.v.len() {
            eprintln!(
                "Expected same # lines in --field <CSV> as mesh, got {} vs {}",
                field.len(),
                m.v.len()
            );
            return Ok(());
        }
        field
    } else {
        (0..m.v.len())
            .map(|_| normalize(std::array::from_fn(|_| rand::random())))
            .collect::<Vec<_>>()
    };

    use pars3d::geom_processing::instant_meshes as im;

    let mut args = im::Args::default();
    args.scale = scale;
    args.orientation_smoothing_iters = 1000;
    args.pos_smoothing_iters = 1000;
    let (new_v, new_f) = im::instant_mesh(&m.v, &m.f, &vn, &mut field, rand::random, &args);

    m.v = new_v;
    m.f = new_f;

    save(dst, &m.into_scene())
}
