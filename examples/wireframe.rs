#![feature(cmp_minmax)]

use pars3d::{load, save};
use std::collections::BTreeSet;

fn main() -> std::io::Result<()> {
    macro_rules! help {
        () => {{
            eprintln!("Outputs the wireframe of <arg1> to <arg2>");
            eprintln!("Usage: <bin> src dst");
            return Ok(());
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

    let scene = load(&src).expect("Failed to load input scene");
    let m = scene.into_flattened_mesh();

    let edges =
        m.f.iter()
            .flat_map(|f| f.edges_ord())
            .collect::<BTreeSet<_>>();

    let wf = pars3d::visualization::colored_wireframe(
        edges.iter().copied(),
        |vi| m.v[vi],
        |[_, _]| [0.; 3],
        1e-3,
    );

    let out_mesh = pars3d::visualization::wireframe_to_mesh(wf);
    let out_scene = out_mesh.into_scene();

    save(dst, &out_scene)
}
