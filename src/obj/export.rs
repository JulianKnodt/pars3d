use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::OutputKind;

use crate::mesh::{Scene, TextureKind};

/// Writes all materials to a single MTL file (with corresponding images)
/// Returns Ok(Some(path to mtl)) on success AND if there is an mtl file.
fn write_mtls(
    s: &Scene,
    obj_file_path: impl AsRef<Path>,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(TextureKind, &str) -> OutputKind,
) -> io::Result<Option<PathBuf>> {
    let mtl_output_kind = if s.mtllibs.is_empty() {
        mtl_file("")
    } else {
        mtl_file(&s.mtllibs[0])
    };

    use crate::util::rel_path_btwn;
    use std::fs::exists;
    let mtl_path = match mtl_output_kind {
        OutputKind::None => return Ok(None),
        OutputKind::ReuseAbsolute
            if s.mtllibs.len() == 1 && exists(&s.mtllibs[0]).unwrap_or(false) =>
        {
            let dst_path: &Path = s.mtllibs[0].as_ref();
            if dst_path.is_absolute() {
                return Ok(Some(PathBuf::from(&s.mtllibs[0])));
            }
            let out_path = dst_path.canonicalize()?;
            return Ok(Some(out_path));
        }
        OutputKind::ReuseRelative
            if s.mtllibs.len() == 1 && exists(&s.mtllibs[0]).unwrap_or(false) =>
        {
            let dst_path: &Path = s.mtllibs[0].as_ref();
            // if original path was absolute don't switch to relative.
            if dst_path.is_absolute() {
                return Ok(Some(PathBuf::from(&s.mtllibs[0])));
            }

            // dst obj -> mtl file
            let out_to_mtl = rel_path_btwn(obj_file_path, dst_path)?;
            return Ok(Some(out_to_mtl));
        }
        OutputKind::ReuseAbsolute | OutputKind::ReuseRelative => {
            panic!("Unable to reuse input MTL file: {}", s.mtllibs[0])
        }
        OutputKind::New(v) => {
            assert_ne!(v, "", "Cannot specify empty new MTL");
            let p: &Path = v.as_ref();
            let obj_path: &Path = obj_file_path.as_ref();
            obj_path.with_file_name(p.file_name().unwrap())
        }
    };

    let mtl_file = File::create(&mtl_path)?;
    let mut mtl_file = BufWriter::new(mtl_file);

    for (mi, mat) in s.materials.iter().enumerate() {
        let mat_name = if !mat.name.is_empty() {
            mat.name.clone()
        } else {
            format!("mat_{mi}")
        };
        writeln!(mtl_file, "newmtl {mat_name}")?;
        for &ti in &mat.textures {
            let tex = &s.textures[ti];
            macro_rules! save_img {
                ($img: expr) => {{
                    let img = $img;
                    match img_dsts(tex.kind, &tex.original_path) {
                        OutputKind::None => continue,
                        OutputKind::ReuseAbsolute
                            if exists(&tex.original_path).unwrap_or(false) =>
                        {
                            let p: &Path = tex.original_path.as_ref();
                            p.canonicalize()?
                        }
                        OutputKind::ReuseRelative
                            if exists(&tex.original_path).unwrap_or(false) =>
                        {
                            rel_path_btwn(&mtl_path, &tex.original_path)?
                        }
                        OutputKind::ReuseAbsolute | OutputKind::ReuseRelative => {
                            panic!("Cannot find original image {}", tex.original_path)
                        }
                        OutputKind::New(f) => {
                            let f: &Path = f.as_ref();
                            let mtl_p: &Path = mtl_path.as_ref();
                            let img_dst = mtl_p.with_file_name(f);
                            match img.save(&mtl_p.with_file_name(f)) {
                                Ok(()) => {}
                                Err(image::ImageError::IoError(err)) => return Err(err),
                                Err(e) => panic!("Failed to save image in OBJ: {e:?}"),
                            }
                            rel_path_btwn(&mtl_path, &img_dst)?
                        }
                    }
                }};
            }
            use crate::mesh::TextureKind;
            macro_rules! write_tex {
                ($tex: expr, $mul_name: expr, $img_name: expr) => {{
                    let tex = $tex;
                    let [r, g, b, _a] = tex.mul;
                    writeln!(mtl_file, "{} {r} {g} {b}", $mul_name)?;
                    let Some(img) = &tex.image else {
                        continue;
                    };
                    let path = save_img!(img);
                    writeln!(mtl_file, "{} {}", $img_name, path.display())?;
                }};
            }
            match tex.kind {
                TextureKind::Diffuse => write_tex!(tex, "Kd", "map_Kd"),
                TextureKind::Specular => write_tex!(tex, "Ks", "map_Ks"),
                TextureKind::Emissive => write_tex!(tex, "Ke", "map_Ke"),
                TextureKind::Normal => {
                    let Some(img) = &tex.image else {
                        continue;
                    };
                    let path = save_img!(img);
                    writeln!(mtl_file, "bump {}", path.display())?;
                }
                // OBJ cannot handle other kinds of textures
                _ => continue,
            }
        }
    }

    Ok(Some(rel_path_btwn(obj_file_path, mtl_path)?))
}

pub fn save_obj(
    s: &Scene,
    geom_path: impl AsRef<Path>,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(TextureKind, &str) -> OutputKind,
) -> io::Result<()> {
    let geom_dst = File::create(&geom_path)?;
    let mut geom_dst = BufWriter::new(geom_dst);

    let has_materials =
        !s.materials.is_empty() && s.materials.iter().any(|m| !m.textures.is_empty());
    let mtl_path = if has_materials {
        write_mtls(s, geom_path, mtl_file, img_dsts)?
    } else {
        None
    };
    //let mtls = s.mtllibs.into_iter().map()
    writeln!(geom_dst, "# Generated by pars3d")?;
    if let Some(mtl_path) = mtl_path {
        writeln!(geom_dst, "mtllib {}", mtl_path.display())?;
    }

    let mut prev_v = 0;
    for (mi, m) in s.meshes.iter().enumerate() {
        for &[x, y, z] in &m.v {
            writeln!(geom_dst, "v {x} {y} {z}")?;
        }

        for &[u, v] in &m.uv[0] {
            writeln!(geom_dst, "vt {u} {v}")?;
        }

        for &[x, y, z] in &m.n {
            writeln!(geom_dst, "vn {x} {y} {z}")?;
        }

        // TODO maybe figure out whether this is correct or not?
        geom_dst.write_all(b"s 1\n")?;
        if m.name.is_empty() {
            writeln!(geom_dst, "g mesh_{mi}")?;
        } else {
            writeln!(geom_dst, "g {}", m.name)?;
        }
        let fmt = |v| match (m.n.is_empty(), m.uv[0].is_empty()) {
            (true, true) => format!("{v}"),
            (true, false) => format!("{v}/{v}"),
            (false, true) => format!("{v}//{v}"),
            (false, false) => format!("{v}/{v}/{v}"),
        };

        // TODO need to write materials here
        let mut curr_mat_idx = None;
        for (fi, f) in m.f.iter().enumerate() {
            match curr_mat_idx {
                None => {
                    if let Some((mat_range, mi)) = m.face_mat_idx.first()
                        && mat_range.contains(&fi)
                    {
                        let name = s.materials.get(*mi).map(|n| n.name.as_str()).unwrap_or("");
                        if name.is_empty() {
                            writeln!(geom_dst, "usemtl mat_{mi}")?;
                        } else {
                            writeln!(geom_dst, "usemtl {name}")?;
                        }
                        curr_mat_idx = Some(0);
                    }
                }
                Some(v) => {
                    let (r, _) = &m.face_mat_idx[v];
                    if !r.contains(&fi) {
                        let next = m.face_mat_idx.iter().position(|v| v.0.contains(&fi));
                        if let Some(ni) = next {
                            curr_mat_idx = Some(ni);
                            let mi = m.face_mat_idx[ni].1;
                            let name = s.materials.get(mi).map(|m| m.name.as_str()).unwrap_or("");
                            if name.is_empty() {
                                writeln!(geom_dst, "usemtl mat_{mi}")?;
                            } else {
                                writeln!(geom_dst, "usemtl {name}")?;
                            }
                        }
                    }
                }
            }
            geom_dst.write_all(b"f ")?;
            let Some((lv, fv)) = f.as_slice().split_last() else {
                continue;
            };
            for &v in fv {
                write!(geom_dst, "{} ", fmt(v + 1 + prev_v))?;
            }
            writeln!(geom_dst, "{}", fmt(lv + 1 + prev_v))?;
        }
        prev_v += m.v.len();
    }
    Ok(())
}

#[test]
fn test_load_save_obj() {
    let scene: crate::mesh::Scene = super::parse("garlic.obj", false, false)
        .expect("Failed to parse obj")
        .into();
    save_obj(
        &scene,
        "tmp/garlic_tmp.obj",
        |_mtl_path| OutputKind::New("tmp.mtl".to_string()),
        |_tex_kind, _og_path| OutputKind::New(format!("{_tex_kind:?}.png")),
    )
    .expect("Failed to save");
}
