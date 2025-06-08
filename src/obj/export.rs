use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::OutputKind;

use crate::mesh::{Scene, TextureKind};

/// Writes all materials to a single MTL file (with corresponding images)
/// Returns Ok(vec![unique paths to mtls]).
fn write_mtls(
    s: &Scene,
    obj_file_path: impl AsRef<Path>,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(TextureKind, &str) -> OutputKind,
) -> io::Result<Vec<PathBuf>> {
    let mut out = vec![];
    // for each material, either append to an existing MTL file or create a new one
    for mat in &s.materials {
        let mtl_output_kind = mtl_file(&mat.path);
        let path = Path::new(&mat.path);

        use crate::util::rel_path_btwn;
        use std::fs::exists;
        let mtl_path = match mtl_output_kind {
            OutputKind::None => continue,
            OutputKind::ReuseAbsolute if exists(&path).unwrap_or(false) => {
                if path.is_absolute() {
                    out.push(PathBuf::from(&path));
                    continue;
                }
                out.push(path.canonicalize()?);
                continue;
            }
            OutputKind::ReuseRelative if exists(&path).unwrap_or(false) => {
                // if original path was absolute don't switch to relative.
                if path.is_absolute() {
                    out.push(PathBuf::from(&path));
                    continue;
                }

                // dst obj -> mtl file
                let out_to_mtl = rel_path_btwn(&obj_file_path, path)?;
                out.push(out_to_mtl);
                continue;
            }
            OutputKind::ReuseAbsolute | OutputKind::ReuseRelative => {
                panic!(
                    "Unable to reuse input MTL file: {}, couldn't find it",
                    path.display()
                );
            }
            OutputKind::New(v) => {
                assert_ne!(v, "", "Cannot specify empty new MTL");
                let p: &Path = v.as_ref();
                let obj_path: &Path = obj_file_path.as_ref();
                obj_path.with_file_name(p.file_name().unwrap())
            }
        };

        let mut mtl_file = File::options();
        mtl_file.write(true).create(true);
        if out.contains(&mtl_path) {
            mtl_file.append(true);
        } else {
            mtl_file.truncate(true);
        }
        let mtl_file = mtl_file.open(&mtl_path).expect("Failed to save mtl");

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
                        let output_kind = img_dsts(tex.kind, &tex.original_path);
                        match output_kind {
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
                                let img_path = mtl_p.with_file_name(f);
                                match img.save(&img_path) {
                                    Ok(()) => {}
                                    Err(image::ImageError::IoError(err)) => {
                                      eprintln!(
                                        "Failed to save output image (should be specified just as file name, got {})",
                                        img_path.display()
                                      );
                                      return Err(err)
                                    },
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
                    TextureKind::Metallic => write_tex!(tex, "Pm", "map_Pm"),
                    TextureKind::Roughness => write_tex!(tex, "Pr", "map_Pr"),
                    TextureKind::Normal => {
                        let Some(img) = &tex.image else {
                            continue;
                        };
                        let path = save_img!(img);
                        writeln!(mtl_file, "bump {}", path.display())?;
                    }
                    // TODO handle more PBR extensions
                    // (https://github.com/tinyobjloader/tinyobjloader/blob/release/pbr-mtl.md)
                    // https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
                }
            }
        }

        out.push(rel_path_btwn(&obj_file_path, mtl_path)?);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub fn save_obj(
    s: &Scene,
    geom_path: impl AsRef<Path>,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(TextureKind, &str) -> OutputKind,
) -> io::Result<()> {
    let geom_dst = File::create(&geom_path)
        .map_err(|e| io::Error::other(format!("Failed to create OBJ file, due to {e:?}")))?;
    let mut geom_dst = BufWriter::new(geom_dst);

    let has_materials =
        !s.materials.is_empty() && s.materials.iter().any(|m| !m.textures.is_empty());
    let mtl_paths = if has_materials {
        write_mtls(s, geom_path, mtl_file, img_dsts)?
    } else {
        vec![]
    };
    //let mtls = s.mtllibs.into_iter().map()
    writeln!(geom_dst, "# Generated by pars3d")?;
    for mtl_path in mtl_paths {
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
