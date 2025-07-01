use pars3d::{load, save};

fn main() {
    let mut src = None;
    let mut dst = None;
    macro_rules! help {
        () => {{
            eprintln!("[HELP]: Import and export a given mesh.");
            eprintln!("Usage: <bin> src dst");
            return;
        }};
    }
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
    println!("[INFO]: {src} -> {dst}");

    let scene = load(&src).expect("Failed to load scene");
    save(dst, &scene).expect("Failed to save scene");
}
