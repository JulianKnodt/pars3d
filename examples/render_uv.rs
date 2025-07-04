#[cfg(feature = "svg")]
use pars3d::{F, load, svg::save_uv};

#[cfg(feature = "svg")]
fn main() {
    macro_rules! help {
        () => {{
            eprintln!("Usage: <bin> <src mesh> dst.svg [-w/--width <float>]");
            return;
        }};
    }

    #[derive(PartialEq, Eq)]
    pub enum State {
        Empty,
        Width,
    }

    let mut state = State::Empty;
    let mut src = None;
    let mut dst = None;
    let mut width = 1.;
    for v in std::env::args().skip(1) {
        match v.as_str() {
            "-w" | "--width" => {
                state = State::Width;
                continue;
            }
            "-h" | "--help" => help!(),
            _ => {}
        }
        match state {
            State::Width => {
                width = v.parse::<F>().unwrap();
                state = State::Empty;
            }
            State::Empty => {
                if src.is_none() {
                    src = Some(v);
                } else if dst.is_none() {
                    dst = Some(v)
                } else {
                    help!();
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

    if src.starts_with("-") || dst.starts_with("-") {
        help!();
    }
    if !dst.ends_with(".svg") {
        help!();
    }
    println!("[INFO]: Rendering UV of {src} to {dst}");

    let scene = load(&src).expect("Failed to load scene");
    let mesh = &scene.meshes[0];
    save_uv(&dst, &mesh.uv[0], &mesh.f, width).expect("Failed to save UVs");
}

#[cfg(not(feature = "svg"))]
fn main() {
    eprintln!("Not compiled with SVG support, not saving");
}
