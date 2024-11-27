use pars3d::{save, load};

fn main() {
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst.is_none() {
            dst = Some(v)
        } else {
            eprintln!("Usage: <bin> src dst");
            return;
        };
    }
    let Some(src) = src else {
        eprintln!("Usage: <bin> src dst");
        return;
    };
    let Some(dst) = dst else {
        eprintln!("Usage: <bin> src dst");
        return;
    };
    println!("[INFO]: {src} -> {dst}");

    let scene = load(&src).expect("Failed to load scene");
    save(dst, &scene).expect("Failed to save scene");
}
