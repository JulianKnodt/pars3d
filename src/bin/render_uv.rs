#[cfg(feature = "svg")]
use pars3d::{load, svg::save_uv};

#[cfg(feature = "svg")]
fn main() {
    macro_rules! help {
        () => {{
            eprintln!("Usage: <bin> <src mesh> dst.svg");
            return;
        }};
    }
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst.is_none() {
            dst = Some(v)
        } else {
            help!();
        };
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
    save_uv(&dst, &mesh.uv[0], &mesh.f, 1.).expect("Failed to save UVs");
}

#[cfg(not(feature = "svg"))]
fn main() {
    eprintln!("Not compiled with SVG support, not saving");
}
