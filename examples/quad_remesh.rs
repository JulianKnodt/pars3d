#[cfg(not(feature = "rand"))]
fn main() {
    eprintln!("Not compiled with rand support");
}

#[cfg(feature = "rand")]
fn main() -> std::io::Result<()> {
    use pars3d::{F, add, cross, divk, kmul, load, normalize, save};
    use std::io::{BufRead, BufReader};

    let mut scale = 0.02;
    let mut field_file = String::new();
    let mut subdivs = 0;
    let mut save_arrows = String::new();
    let mut save_grid = String::new();
    let mut rotate_90 = false;
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
              [--rotate-90 = false]"#);
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
    let og_verts = m.v.clone();
    let new_vert_map = m.geometry_only();
    m.triangulate();
    m.f.retain_mut(|f| !f.canonicalize());
    for v in &mut m.v {
        v.swap(0, 2);
        v[2] = -v[2];
        v[0] = -v[0];
    }
    let (s, t) = m.normalize();

    let mut vn = vec![];
    pars3d::geom_processing::vertex_normals(&m.f, &m.v, &mut vn, Default::default());
    let mut zero_normals = vn
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == [0.; 3])
        .map(|i_n| i_n.0)
        .collect::<Vec<_>>();

    let vv_adj = m.vertex_vertex_adj().uniform();
    let mut ri = 0;
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
        if total_w == 0. {
            zero_normals.push(vi);
            let l = zero_normals.len();
            // always swap with a different element so it never gets totally stuck.
            zero_normals.swap(l - 1, ri % l);
            ri = (ri + 1) % l;
            continue;
        }
        vn[vi] = normalize(ns);
    }

    let mut field = if !field_file.is_empty() {
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
            let mut new_field = vec![[0.; 3]; m.v.len()];
            let mut ws = vec![0; m.v.len()];
            for (vi, v) in og_verts.into_iter().enumerate() {
                let new_idx = new_vert_map[&v.map(F::to_bits)];
                let old_field = field[vi];
                new_field[new_idx] = add(new_field[new_idx], old_field);
                ws[new_idx] += 1;
            }
            for (vi, f) in new_field.iter_mut().enumerate() {
                assert_ne!(ws[vi], 0);
                *f = divk(*f, ws[vi] as F);
            }
            field = new_field;
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

    use pars3d::geom_processing::instant_meshes as im;

    let mut args = im::Args::default();
    args.scale = scale;
    args.orientation_smoothing_iters = if field_file.is_empty() { 100 } else { 10 };
    args.pos_smoothing_iters = 100;
    args.save_grid = save_grid;
    let (new_v, new_f) = im::instant_mesh(&m.v, &m.f, &vn, &mut field, rand::random, &args);

    if !save_arrows.is_empty() {
        let (v, c, f) = pars3d::visualization::arrows(
            &m.v,
            &field,
            0.1,
            Some([[1., 0.86, 0.], [0.25, 0.05, 0.05]]),
        );
        let f = f
            .into_iter()
            .map(pars3d::FaceKind::Quad)
            .collect::<Vec<_>>();
        let mut wf = pars3d::Mesh::new_geometry(v, f);
        wf.vert_colors = c;
        //wf.denormalize(s,t);
        save(save_arrows, &wf.into_scene())?;
    }

    m.v = new_v;
    m.f = new_f;

    m.denormalize(s, t);
    save(dst, &m.into_scene())
}
