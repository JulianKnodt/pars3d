use pars3d::{load, save};

fn main() {
    let mut src = None;
    let mut dst = None;
    let mut triangulate = false;
    let mut option_list: [(_, Option<&str>, &mut bool, &str); _] = [(
        "--triangulate",
        None,
        &mut triangulate,
        "Triangulate output mesh",
    )];
    macro_rules! help {
        () => {{
            eprintln!("[HELP]: \nImport and export a given mesh.");
            eprintln!("Basic Usage: <bin> src dst");
            for (l, s, _, help) in option_list {
                if let Some(s) = s {
                    eprintln!("\t {s}, {l} : {help}");
                } else {
                    eprintln!("\t {l} : {help}");
                }
            }
            return;
        }};
    }
    for v in std::env::args().skip(1) {
        if let Some(opt) = option_list
            .iter_mut()
            .find(|opt| opt.0 == v || opt.1.is_some_and(|short| short == v))
        {
            *opt.2 = true;
            continue;
        }
        if matches!(v.as_str(), "-h" | "--help") {
            help!();
        }

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

    let mut scene = load(&src).expect("Failed to load scene");
    if triangulate {
        for m in &mut scene.meshes {
            m.triangulate(0);
        }
    }
    save(dst, &scene).expect("Failed to save scene");
}
