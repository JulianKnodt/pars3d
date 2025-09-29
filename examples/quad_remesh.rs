#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, add, cross, dist, dot, kmul, load, normalize, save, sub};
    use std::io::{BufRead, BufReader};

    let mut scale = 0.02;
    let mut field_file = String::new();
    let mut subdivs = 0;
    let mut save_arrows = String::new();
    let mut save_grid = String::new();
    let mut rotate_90 = false;
    let mut use_color_field = false;
    let mut orient_iters = 100;
    macro_rules! help {
        ($( $err: tt )?) => {{
            $(eprintln!("[ERROR]: {}", format!($err));)*
            eprintln!("[]: Quad Remeshes <arg1> to <arg2>");
            eprintln!(r#"Usage: <bin> input(#0) dst(#1)
              [-i, --input, #0 <SRC> REQUIRED]
              [-o, --output, #1 <SRC> REQUIRED]
              [--scale <float> = {scale}]
              [--field <CSV> = None]
              [--subdivisions <INT> = {subdivs}]
              [--save-arrows <DST> = None]"
              [--save-grid <DST> = None]
              [--stats <DST> = None]
              [--rotate-90 = false]
              [--color-field = false]
              [--orient-iters <INT> = {orient_iters}]"#);
            return Ok(());
        }};
    }

    #[derive(PartialEq, Eq, Debug)]
    pub enum State {
        Empty,
        Scale,
        Field,
        Subdiv,
        SaveArrows,
        SaveGrid,
        Input,
        Output,
        Stats,
        OrientIters,
    }

    let mut state = State::Empty;
    let mut src = None;
    let mut dst = None;
    for v in std::env::args().skip(1) {
        match v.as_str() {
            "-h" | "--help" => help!(),
            "--scale" => {
                state = State::Scale;
                continue;
            }
            "--field" => {
                state = State::Field;
                continue;
            }
            "--subdivisions" => {
                state = State::Subdiv;
                continue;
            }
            "--save-arrows" => {
                state = State::SaveArrows;
                continue;
            }
            "--save-grid" => {
                state = State::SaveGrid;
                continue;
            }
            "-i" | "--input" => {
                state = State::Input;
                continue;
            }
            "-o" | "--output" => {
                state = State::Output;
                continue;
            }
            "--stats" => {
                state = State::Stats;
                continue;
            }
            "--rotate-90" => {
                rotate_90 = true;
                continue;
            }
            "--color-field" => {
                use_color_field = true;
                continue;
            }
            "--orient-iters" => {
                state = State::OrientIters;
                continue;
            }
            v if v.starts_with('-') => help!("Unknown flag {v}"),
            _ => {}
        };
        match state {
            State::Scale => {
                scale = match v.parse::<F>() {
                    Ok(s) => s,
                    Err(e) => help!("Failed to parse scale ({v:?}) as float, err: {e}"),
                };
            }
            State::Subdiv => {
                subdivs = match v.parse::<usize>() {
                    Ok(s) => s,
                    Err(e) => help!("Failed to parse subdivs ({v:?}) as usize, err: {e}"),
                };
            }
            State::OrientIters => {
                orient_iters = match v.parse::<usize>() {
                    Ok(s) => s,
                    Err(e) => help!("Failed to parse orient iters ({v:?}) as usize, err: {e}"),
                };
            }
            State::Field => {
                field_file = v;
            }
            State::SaveArrows => {
                save_arrows = v;
            }
            State::SaveGrid => {
                save_grid = v;
            }
            State::Input => {
                src = Some(v);
            }
            State::Output => {
                dst = Some(v);
            }
            State::Stats => {
                // ... Unused
            }
            State::Empty => {
                if src.is_none() {
                    src = Some(v);
                } else if dst.is_none() {
                    dst = Some(v)
                } else {
                    help!("Too many arguments (got {v})");
                };
            }
        }
        state = State::Empty;
    }
    if state != State::Empty {
        help!("Passed a flag, but did not get parameter ({state:?})");
    }
    let Some(src) = src else {
        help!("Missing source & dest paths");
    };
    let Some(dst) = dst else {
        help!("Missing dest path");
    };

    let scene = load(&src).expect("Failed to load input scene");
    let mut m = scene.into_flattened_mesh();

    let mut color_field = if use_color_field {
        vec![[0.; 3]; m.v.len()]
    } else {
        vec![]
    };
    for f in m.f.iter() {
        if !use_color_field {
            break;
        }
        for vis @ [vi0, vi1] in f.all_pairs_ord() {
            let [c0, c1] = vis.map(|vi| m.vert_colors[vi]);
            let [p0, p1] = vis.map(|vi| m.v[vi]);

            fn luma(rgb: [F; 3]) -> F {
                dot(rgb, [0.299, 0.587, 0.114])
            }
            let delta = (luma(c0) - luma(c1)).abs() / dist(p0, p1).max(1e-5);
            let e = kmul(delta, sub(p0, p1));
            color_field[vi0] = add(color_field[vi0], e);
            color_field[vi1] = add(color_field[vi1], e.map(core::ops::Neg::neg));
        }
    }

    let og_verts = m.v.clone();
    let new_vert_map = m.geometry_only();
    /*
    for v in &mut m.v {
        v.swap(0, 2);
        v[2] = -v[2];
        v[0] = -v[0];
    }
    */
    let (s, t) = m.normalize();
    m.triangulate(0);
    m.f.retain_mut(|f| !f.canonicalize());

    let mut vn = vec![];
    pars3d::geom_processing::vertex_normals(&m.f, &m.v, &mut vn, Default::default());
    let mut zero_normals = vn
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == [0.; 3])
        .map(|i_n| i_n.0)
        .collect::<Vec<_>>();

    let vv_adj = m.vertex_vertex_adj().laplacian(&m.f, &m.v);
    let mut ri = 0;
    assert_ne!(zero_normals.len(), m.v.len());

    // stupid way to spread normals to where there are zero values
    while let Some(vi) = zero_normals.pop() {
        let mut total_w = 0.;
        let mut ns = [0.; 3];
        for (adj_vi, w) in vv_adj.adj_data(vi) {
            let adj_n = vn[adj_vi as usize];
            if adj_n == [0.; 3] {
                continue;
            }
            total_w += w;
            ns = add(ns, kmul(w, adj_n));
        }
        if total_w == 0. || pars3d::length(ns) == 0. {
            zero_normals.push(vi);
            let l = zero_normals.len();
            // always swap with a different element so it never gets totally stuck.
            zero_normals.swap(l - 1, ri % l);
            ri = (ri + 1) % l;
            continue;
        }
        vn[vi] = normalize(ns);
    }

    let mut field = if use_color_field {
        if !field_file.is_empty() {
            eprintln!("Please specify only one of `--color-field` or `--field`.");
            return Ok(());
        }
        m.copy_attribs(&og_verts, new_vert_map, &color_field)
        /*
        let avg = new_field.iter().copied().map(pars3d::length).sum::<F>() / new_field.len() as F;
        let mut total_reset = 0;
        for v in new_field.iter_mut() {
          if pars3d::length(*v) < 1.5 * avg {
            *v = normalize(std::array::from_fn(|_| rand::random()));
            total_reset += 1;
          }
        }
        println!("{total_reset}/{}", new_field.len());
        */
    } else if !field_file.is_empty() {
        let f = std::fs::File::open(&field_file)?;
        let mut field = vec![];
        for l in BufReader::new(f).lines() {
            let l = l?;
            let mut keys = l.split(",");
            let mut vec = [0.; 3];
            for i in 0..3 {
                let Some(k) = keys.next() else {
                    eprintln!("Not enough values in --field <CSV> (expected 3 per line, got {i})");
                    return Ok(());
                };
                let Ok(k) = k.parse::<F>() else {
                    eprintln!("Expected numerical values in --field <CSV>, got {k}");
                    return Ok(());
                };
                vec[i] = k;
            }
            field.push(vec);
        }
        // correct if geometry consolidation caused the field to be a different size
        if og_verts.len() != m.v.len() && field.len() == og_verts.len() {
            field = m.copy_attribs(&og_verts, new_vert_map, &field);
        }

        if field.len() != m.v.len() {
            eprintln!(
                "Expected same # lines in --field <CSV> as mesh, got {} in field vs {} in mesh",
                field.len(),
                m.v.len()
            );
            return Ok(());
        }
        println!("[INFO]: Loaded field successfully from {field_file}");

        let mut zero_fields = field
            .iter()
            .enumerate()
            .filter(|(_, f)| **f == [0.; 3])
            .map(|i_f| i_f.0)
            .collect::<Vec<_>>();

        let mut ri = 0;
        // stupid way to spread normals to where there are zero values
        while let Some(vi) = zero_fields.pop() {
            let mut total_w = 0.;
            let mut fs = [0.; 3];
            for (adj_vi, w) in vv_adj.adj_data(vi) {
                let adj_f = field[adj_vi as usize];
                if adj_f == [0.; 3] {
                    continue;
                }
                total_w += w;
                fs = add(fs, kmul(w, adj_f));
            }
            if total_w == 0. {
                zero_fields.push(vi);
                let l = zero_fields.len();
                // always swap with a different element so it never gets totally stuck.
                zero_fields.swap(l - 1, ri % l);
                ri = (ri + 1) % l;
                continue;
            }
            field[vi] = normalize(fs);
        }

        field
    } else {
        (0..m.v.len())
            .map(|_| normalize(std::array::from_fn(|_| rand::random())))
            .collect::<Vec<_>>()
    };

    if rotate_90 {
        for (vi, v) in &mut field.iter_mut().enumerate() {
            *v = cross(vn[vi], *v);
        }
    }

    use pars3d::geom_processing::subdivision;
    let mut tris =
        m.f.drain(..)
            .map(|t| t.as_tri().unwrap())
            .collect::<Vec<_>>();
    let mut barys = vec![];
    for _ in 0..subdivs {
        let (bary, new_tris) = subdivision::loop_subdivision(&tris);
        barys.push(bary);
        tris = new_tris;
    }

    m.f.extend(tris.into_iter().map(pars3d::FaceKind::Tri));

    if !barys.is_empty() {
        let new_bary = barys
            .into_iter()
            .reduce(|p, n| subdivision::compose_barycentric_repr(&n, &p).collect::<Vec<_>>())
            .unwrap();
        m.v = new_bary.iter().map(|b| b.eval(&m.v)).collect();
        field = new_bary.iter().map(|b| b.eval(&field)).collect();
        vn = new_bary.iter().map(|b| normalize(b.eval(&vn))).collect();
    }
    if subdivs != 0 {
        println!(
            "[INFO]: Subdivided to {} vertices and {} faces.",
            m.v.len(),
            m.f.len()
        );
    } else {
        println!(
            "[INFO]: Input has {} vertices and {} faces",
            m.v.len(),
            m.f.len()
        );
    }
    //pars3d::save(&"ref.obj", &m.clone().into_scene());

    if !save_arrows.is_empty() {
        let mut wf = pars3d::Mesh::new_geometry(m.v.clone(), m.f.clone());
        wf.vert_colors = field
            .iter()
            .map(|&d| kmul(0.5, add(d, [1.; 3])))
            .collect::<Vec<_>>();
        wf.denormalize(s, t);
        save(&save_arrows, &wf.into_scene())?;
    }

    use pars3d::geom_processing::instant_meshes as im;

    let mut args = im::Args::default();
    args.s = s;
    args.t = t;
    args.scale = scale;
    args.dissolve_edges = true;
    args.orientation_smoothing_iters = orient_iters;
    args.pos_smoothing_iters = 100;
    args.save_grid = save_grid;

    args.locked = if use_color_field {
        let avg = field.iter().copied().map(pars3d::length).sum::<F>() / field.len() as F;
        let locked = field
            .iter()
            .map(|&f| pars3d::length(f) > 2.4 * avg)
            .collect::<Vec<_>>();
        for (i, &l) in locked.iter().enumerate() {
            if l {
                continue;
            }
            field[i] = normalize(std::array::from_fn(|_| rand::random()));
        }
        locked
        //vec![]
    } else {
        vec![]
    };
    if !args.locked.is_empty() {
        eprintln!(
            "[INFO]: Locked {}/{} vertices",
            args.locked.iter().filter(|v| **v).count(),
            field.len()
        );
    }
    let (new_v, new_f) = im::instant_mesh(&m.v, &m.f, &vn, &mut field, &args);

    if !save_arrows.is_empty() {
        let mut wf = pars3d::Mesh::new_geometry(m.v, m.f);
        wf.vert_colors = field
            .iter()
            .map(|&d| kmul(0.5, add(d, [1.; 3])))
            .collect::<Vec<_>>();
        /*
        let (v, c, f) = pars3d::visualization::arrows(
            &m.v,
            &field,
            0.01,
            Some([[1., 0.86, 0.], [0.25, 0.05, 0.05]]),
        );
        let f = f
            .into_iter()
            .map(pars3d::FaceKind::Quad)
            .collect::<Vec<_>>();
        let mut wf = pars3d::Mesh::new_geometry(v, f);
        wf.vert_colors = c;
        */
        wf.denormalize(s, t);
        save(&save_arrows, &wf.into_scene())?;
    }

    m.v = new_v;
    m.f = new_f;

    m.denormalize(s, t);
    save(dst, &m.into_scene())
}
