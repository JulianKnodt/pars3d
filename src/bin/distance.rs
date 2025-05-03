#[cfg(not(feature = "kdtree"))]
fn main() {
    eprintln!(
        r#"
  _,    _,   ,_    _,
 / \,  / \,  |_)  (_,
'\_/  '\_/  '|     _)
 '     '     '    '"#
    );
    eprintln!("Not compiled with KDTree support, exiting.");
}

#[cfg(feature = "kdtree")]
fn main() {
    use kdtree::KDTree;
    use pars3d::{dist, length, load, F};

    macro_rules! help {
        ($( $extra: expr )?) => {{
            $(
              eprintln!($extra);
            )?
            eprintln!(
                r#"Usage: <bin> <mesh a> <mesh b> <#samples:int>
- A tool for evaluating the distance between two meshes including their attributes."#
            );
            return;
        }};
    }
    let mut src = None;
    let mut dst = None;
    let mut num_samples = None;
    for v in std::env::args().skip(1) {
        let mut any = false;
        for arg in [&mut src, &mut dst, &mut num_samples] {
            if arg.is_some() {
                continue;
            }
            any = true;
            *arg = Some(v);
            break;
        }
        if !any {
            help!("Got too many arguments, expected 3 or 4 arguments");
        };
    }
    let Some(src) = src else {
        help!("Missing 1st argument, <mesh a>");
    };
    let Some(dst) = dst else {
        help!("Missing 2nd argument, <mesh b>");
    };
    if src.starts_with("-") || dst.starts_with("-") {
        help!("No flags are supported, assuming help");
    }
    let Some(num_samples) = num_samples.as_ref().and_then(|ns| ns.parse::<usize>().ok()) else {
        help!("Did not get number of samples to add, instead got {num_samples:?}");
    };

    println!("[INFO]: Evaluating geometric distance between {src} & {dst}");

    let a = load(&src)
        .expect(&format!("Failed to load {}", src))
        .into_flattened_mesh();
    let b = load(&dst)
        .expect(&format!("Failed to load {}", dst))
        .into_flattened_mesh();

    let a_aabb = a.aabb();
    let b_aabb = b.aabb();

    let diag_len = length(a_aabb.diag()).max(length(b_aabb.diag()));
    assert!(diag_len > 0., "Both meshes are degenerate");

    let mut i = a.v.len() as F;
    let a_samples = a
        .random_points_on_mesh(num_samples, || {
            let v = (i * 389.21 + 0.348).cos();
            i += v;
            v.fract()
        })
        .map(|(fi, b)| a.f[fi].map_kind(|vi| a.v[vi]).from_barycentric(b));
    let a_kdtree = KDTree::<(), 3, false, F>::new(
        a.v.iter().copied().chain(a_samples).map(|v| (v, ())),
        Default::default(),
    );

    let mut i = b.v.len() as F;
    let b_samples = b
        .random_points_on_mesh(num_samples, || {
            let v = (i * 389.21 + 0.348).cos();
            i += v;
            v.fract()
        })
        .map(|(fi, bary)| b.f[fi].map_kind(|vi| b.v[vi]).from_barycentric(bary));
    let b_kdtree = KDTree::<(), 3, false, F>::new(
        b.v.iter().copied().chain(b_samples).map(|v| (v, ())),
        Default::default(),
    );

    let mut a_to_b_dists = vec![];
    for &p in a_kdtree.points() {
        let (&nearest, _d, ()) = b_kdtree.nearest(&p).unwrap();
        a_to_b_dists.push(dist(p, nearest) / diag_len);
    }

    let mut b_to_a_dists = vec![];
    for &p in b_kdtree.points() {
        let (&nearest, _d, ()) = a_kdtree.nearest(&p).unwrap();
        b_to_a_dists.push(dist(p, nearest) / diag_len);
    }

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

    println!(
        r#"{{
  "hausdorff": {hausdorff},
  "chamfer": {chamfer},
  "hausdorff_a_to_b": {hausdorff_a_to_b},
  "hausdorff_b_to_a": {hausdorff_b_to_a},
  "chamfer_a_to_b": {chamfer_a_to_b},
  "chamfer_b_to_a": {chamfer_b_to_a}
}}"#
    );
}
