#![feature(cmp_minmax)]

use pars3d::{F, load, save};
use std::collections::BTreeSet;

fn main() -> std::io::Result<()> {
    let mut width = 1.5e-3;
    macro_rules! help {
        () => {{
            eprintln!("Outputs the wireframe of <arg1> to <arg2>");
            eprintln!("Usage: <bin> src dst [-w/--width <float> (default = {width})]");
            return Ok(());
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
        width,
    );

    let out_mesh = pars3d::visualization::wireframe_to_mesh(wf);
    let out_scene = out_mesh.into_scene();

    save(dst, &out_scene)
}
