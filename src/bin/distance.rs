#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[cfg(all(feature = "kdtree", feature = "rand"))]
use kdtree::KDTree;
#[cfg(all(feature = "kdtree", feature = "rand"))]
use pars3d::{length, load, F};

#[cfg(not(all(feature = "kdtree", feature = "rand")))]
fn main() {
    eprintln!(
        r#"
  _,    _,   ,_    _,
 / \,  / \,  |_)  (_,
'\_/  '\_/  '|     _)
 '     '     '    '"#
    );
    eprintln!("Not compiled with KDTree & rand support, exiting.");
}

#[cfg(all(feature = "kdtree", feature = "rand"))]
fn main() -> std::io::Result<()> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ParseState {
        Default,
        TextureA,
        TextureB,
        Stat,
    }

    macro_rules! help {
        ($( $extra: expr )?) => {{
            $(
              eprintln!($extra);
            )?
            eprintln!(
                r#"Usage: <bin> <mesh a> <mesh b> <#samples:int>
  <--with-color> <--texture-a [PATH]> <--texture-b [PATH]> <--stat [PATH].json>
- A tool for evaluating the distance between two meshes including their attributes."#
            );
            return Ok(());
        }};
    }
    let mut src = None;
    let mut dst = None;
    let mut num_samples = None;
    let mut with_color = false;
    let mut texture_a = None;
    let mut texture_b = None;
    let mut stat = None;

    let mut state = ParseState::Default;
    let mut default_to_fill = [&mut src, &mut dst, &mut num_samples];
    let mut tex_a_to_fill = [&mut texture_a];
    let mut tex_b_to_fill = [&mut texture_b];
    let mut stat_to_fill = [&mut stat];

    for v in std::env::args().skip(1) {
        if v == "-h" || v == "--help" {
            help!();
        }

        if v == "--with-color" {
            with_color = true;
            continue;
        } else if v == "--texture-a" {
            state = ParseState::TextureA;
            continue;
        } else if v == "--texture-b" {
            state = ParseState::TextureB;
            continue;
        } else if v == "--stat" {
            state = ParseState::Stat;
            continue;
        }

        let to_fill = match state {
            ParseState::Default => default_to_fill.as_mut_slice(),
            ParseState::TextureA => tex_a_to_fill.as_mut_slice(),
            ParseState::TextureB => tex_b_to_fill.as_mut_slice(),
            ParseState::Stat => stat_to_fill.as_mut_slice(),
        };

        let Some(dst) = to_fill.iter_mut().find(|v| v.is_none()) else {
            help!("Got unknown argument {v}");
        };
        **dst = Some(v);

        state = ParseState::Default;
    }
    if state != ParseState::Default {
        help!("Missing path to texture ({state:?})");
    }
    let Some(src) = src else {
        help!("Missing 1st argument, <mesh a>");
    };
    let Some(dst) = dst else {
        help!("Missing 2nd argument, <mesh b>");
    };
    if src.starts_with("-") {
        help!("Unknown flag {src:?}, assuming help");
    }
    if dst.starts_with("-") {
        help!("Unknown flag {dst:?}, assuming help");
    }
    const DEFAULT_NUM_SAMPLES: usize = 500000;
    let num_samples = match num_samples {
        None => {
            println!("[INFO]: Using default number of samples {DEFAULT_NUM_SAMPLES}");
            DEFAULT_NUM_SAMPLES
        }
        Some(ns) => {
            if let Ok(num_samples) = ns.parse::<usize>() {
                num_samples
            } else {
                help!("Did not get samples as #, instead got {ns:?}");
            }
        }
    };

    let mut a = load(&src)
        .expect(&format!("Failed to load {}", src))
        .into_flattened_mesh();
    a.triangulate();
    let mut b = load(&dst)
        .expect(&format!("Failed to load {}", dst))
        .into_flattened_mesh();
    b.triangulate();

    let a_aabb = a.aabb();
    let b_aabb = b.aabb();

    let diag_len = length(a_aabb.diag()).max(length(b_aabb.diag()));
    assert!(diag_len > 0., "Both meshes are degenerate");

    let metrics = if with_color {
        geometric_texture_distance(&a, &b, diag_len, num_samples, texture_a, texture_b)
    } else {
        geometric_distance(&a, &b, diag_len, num_samples)
    };

    for (k, v) in metrics {
        println!("{k}: {v}");
    }

    use std::io::Write;
    if let Some(stat) = stat.as_ref() {
        if std::fs::exists(stat)? {
            let mut s = std::fs::read_to_string(stat)?;
            for (k, v) in metrics {
                pars3d::util::append_json(&mut s, 2, k, v);
            }
            std::fs::write(stat, &s)?;
        } else {
            let mut f = std::fs::File::create(stat)?;
            write!(f, "{{\n")?;
            for (i, (k, v)) in metrics.into_iter().enumerate() {
                write!(f, "  \"{k}\": {v}")?;
                writeln!(f, "{}", if i == metrics.len() - 1 { "" } else { "," })?;
            }
            writeln!(f, "}}")?;
        }
    }

    Ok(())
}

#[cfg(all(feature = "kdtree", feature = "rand"))]
fn geometric_distance(
    a: &pars3d::Mesh,
    b: &pars3d::Mesh,
    diag_len: F,
    num_samples: usize,
) -> [(&'static str, f64); 6] {
    let a_samples = a
        .random_points_on_mesh(num_samples, rand::random)
        .map(|(fi, b)| a.f[fi].map_kind(|vi| a.v[vi]).from_barycentric(b));
    let a_kdtree = KDTree::<(), 3, false, F>::new(
        a.v.iter().copied().chain(a_samples).map(|v| (v, ())),
        Default::default(),
    );

    let b_samples = b
        .random_points_on_mesh(num_samples, rand::random)
        .map(|(fi, bary)| b.f[fi].map_kind(|vi| b.v[vi]).from_barycentric(bary));
    let b_kdtree = KDTree::<(), 3, false, F>::new(
        b.v.iter().copied().chain(b_samples).map(|v| (v, ())),
        Default::default(),
    );

    let a_to_b_dists = a_kdtree
        .points()
        .iter()
        .map(|p| b_kdtree.nearest(p).unwrap().1 / diag_len)
        .collect::<Vec<_>>();

    let b_to_a_dists = b_kdtree
        .points()
        .iter()
        .map(|p| a_kdtree.nearest(p).unwrap().1 / diag_len)
        .collect::<Vec<_>>();

    let hausdorff_a_to_b = a_to_b_dists
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(&b).unwrap())
        .unwrap();
    let chamfer_a_to_b = a_to_b_dists.iter().copied().sum::<F>() / a_to_b_dists.len() as F;

    let hausdorff_b_to_a = b_to_a_dists
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(&b).unwrap())
        .unwrap();
    let chamfer_b_to_a = b_to_a_dists.iter().copied().sum::<F>() / b_to_a_dists.len() as F;

    let hausdorff = hausdorff_a_to_b.max(hausdorff_b_to_a);
    let chamfer = chamfer_a_to_b + chamfer_b_to_a;

    [
        ("hausdorff", hausdorff),
        ("chamfer", chamfer),
        ("hausdorff_a_to_b", hausdorff_a_to_b),
        ("hausdorff_b_to_a", hausdorff_b_to_a),
        ("chamfer_a_to_b", chamfer_a_to_b),
        ("chamfer_b_to_a", chamfer_b_to_a),
    ]
}

/// Compute the geometric distance between two meshes, either with texture or vertex colors.
#[cfg(all(feature = "kdtree", feature = "rand"))]
fn geometric_texture_distance(
    a: &pars3d::Mesh,
    b: &pars3d::Mesh,
    diag_len: F,
    num_samples: usize,
    texture_a: Option<String>,
    texture_b: Option<String>,
) -> [(&'static str, f64); 6] {
    const CHAN: usize = 0;
    use image::imageops::{flip_vertical, sample_bilinear};
    let tex_a = if let Some(ta) = texture_a {
        assert_eq!(a.uv[CHAN].len(), a.v.len(), "Missing UV on mesh a");
        Some(flip_vertical(
            &image::open(ta).expect("Failed to open texture a"),
        ))
    } else {
        assert_eq!(
            a.v.len(),
            a.vert_colors.len(),
            "Texture required for <mesh a> w/o vertex colors"
        );
        None
    };
    let tex_b = if let Some(tb) = texture_b {
        assert_eq!(b.uv[CHAN].len(), b.v.len(), "Missing UV on mesh b");
        Some(flip_vertical(
            &image::open(tb).expect("Failed to open texture b"),
        ))
    } else {
        assert_eq!(
            b.v.len(),
            b.vert_colors.len(),
            "Texture required for mesh b w/o vertex colors"
        );
        None
    };
    fn concat<const N: usize, const M: usize>(a: [F; N], b: [F; M]) -> [F; N + M] {
        std::array::from_fn(|i| if i < N { a[i] } else { b[i - N] })
    }
    let a_samples = a
        .random_points_on_mesh(num_samples, rand::random)
        .map(|(fi, b)| {
            let f = &a.f[fi];
            let p = f.map_kind(|vi| a.v[vi]).from_barycentric(b);
            let c = if let Some(tex_a) = tex_a.as_ref() {
                let [u, v] = f
                    .map_kind(|vi| a.uv[CHAN][vi])
                    .from_barycentric(b)
                    .map(|v| v.fract().abs());
                let image::Rgba([r, g, b, _a]) =
                    sample_bilinear(tex_a, u as f32, v as f32).unwrap();
                [r, g, b].map(|v| v as F / 255.)
            } else {
                f.map_kind(|vi| a.vert_colors[vi]).from_barycentric(b)
            };
            concat(p, c)
        });
    let verts = (0..a.v.len()).map(|i| {
        let c = if let Some(tex_a) = tex_a.as_ref() {
            let [u, v] = a.uv[CHAN][i];
            let image::Rgba([r, g, b, _a]) = sample_bilinear(tex_a, u as f32, v as f32).unwrap();
            [r, g, b].map(|v| v as F / 255.)
        } else {
            a.vert_colors[i]
        };
        concat(a.v[i], c)
    });
    let a_kdtree =
        KDTree::<(), 6, false, F>::new(verts.chain(a_samples).map(|v| (v, ())), Default::default());

    let b_samples = b
        .random_points_on_mesh(num_samples, rand::random)
        .map(|(fi, bary)| {
            let f = &b.f[fi];
            let p = f.map_kind(|vi| b.v[vi]).from_barycentric(bary);
            let c = if let Some(tex_b) = tex_b.as_ref() {
                let [u, v] = f
                    .map_kind(|vi| b.uv[CHAN][vi])
                    .from_barycentric(bary)
                    .map(|v| v.fract().abs());
                let image::Rgba([r, g, b, _a]) =
                    sample_bilinear(tex_b, u as f32, v as f32).unwrap();
                [r, g, b].map(|v| v as F / 255.)
            } else {
                f.map_kind(|vi| b.vert_colors[vi]).from_barycentric(bary)
            };
            concat(p, c)
        });
    let verts = (0..b.v.len()).map(|i| {
        let c = if let Some(tex_b) = tex_b.as_ref() {
            let [u, v] = b.uv[CHAN][i];
            let image::Rgba([r, g, b, _a]) = sample_bilinear(tex_b, u as f32, v as f32).unwrap();
            [r, g, b].map(|v| v as F / 255.)
        } else {
            b.vert_colors[i]
        };
        concat(b.v[i], c)
    });
    let b_kdtree =
        KDTree::<(), 6, false, F>::new(verts.chain(b_samples).map(|v| (v, ())), Default::default());

    let mut max_a_to_b = 0. as F;
    let mut sum_a_to_b = 0.;
    let mut err_a_to_b = 0.;
    for p in a_kdtree.points().iter() {
        let d = b_kdtree.nearest(p).unwrap().1 / diag_len;
        max_a_to_b = max_a_to_b.max(d);

        // kahan sum
        let y = d - err_a_to_b;
        let t = sum_a_to_b + y;
        err_a_to_b = (t - sum_a_to_b) - y;
        sum_a_to_b = t;
    }
    let hausdorff_a_to_b = max_a_to_b;
    let chamfer_a_to_b = sum_a_to_b / a_kdtree.points().len() as F;

    let mut max_b_to_a = 0. as F;
    let mut sum_b_to_a = 0.;
    let mut err_b_to_a = 0.;
    for p in b_kdtree.points().iter() {
        let d = b_kdtree.nearest(p).unwrap().1 / diag_len;
        max_b_to_a = max_b_to_a.max(d);

        // kahan sum
        let y = d - err_b_to_a;
        let t = sum_b_to_a + y;
        err_b_to_a = (t - sum_b_to_a) - y;
        sum_b_to_a = t;
    }

    let hausdorff_b_to_a = max_b_to_a;
    let chamfer_b_to_a = sum_b_to_a / b_kdtree.points().len() as F;

    let hausdorff = hausdorff_a_to_b.max(hausdorff_b_to_a);
    let chamfer = chamfer_a_to_b + chamfer_b_to_a;
    [
        ("hausdorff", hausdorff),
        ("chamfer", chamfer),
        ("hausdorff_a_to_b", hausdorff_a_to_b),
        ("hausdorff_b_to_a", hausdorff_b_to_a),
        ("chamfer_a_to_b", chamfer_a_to_b),
        ("chamfer_b_to_a", chamfer_b_to_a),
    ]
}
