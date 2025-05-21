use pars3d::load;
use std::io::Write;

fn main() -> std::io::Result<()> {
    macro_rules! help {
        () => {{
            eprintln!("Usage: <bin> src [OPTIONAL dst_json]");
            return Ok(());
        }};
    }
    let mut src = None;
    let mut dst_json = None;
    for v in std::env::args().skip(1) {
        if src.is_none() {
            src = Some(v);
        } else if dst_json.is_none() {
            dst_json = Some(v)
        } else {
        };
    }
    let Some(src) = src else {
        help!();
    };

    let scene = load(&src).expect("Failed to load input scene");
    let mut has_bd = false;
    let mut has_nm = false;
    for m in &scene.meshes {
        let (_, bd_e, nm_e) = m.num_edge_kinds();
        has_bd = has_bd || bd_e > 0;
        has_nm = has_nm || nm_e > 0;
    }
    println!("[INFO]: has boundaries {has_bd}, has non-manifold {has_nm}");
    let Some(dst_json) = dst_json else {
        return Ok(());
    };
    let mut f = std::fs::File::create(&dst_json).expect(&format!("Failed to create {dst_json}"));
    write!(f, r#"{{ "has_bd": {has_bd}, "has_nm": {has_nm} }}"#)
}
