use super::{Vec2, Vec3, F};

use image::DynamicImage;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use std::ffi::OsStr;

pub type FaceIdx = usize;
pub type VertIdx = usize;
pub type VertNIdx = usize;
pub type VertTIdx = usize;
pub type MatIdx = usize;

/// A single triangular mesh face.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MeshFace {
    pub v: [usize; 3],
    pub vt: Option<[usize; 3]>,
    pub vn: Option<[usize; 3]>,
}

/// A single polygonal mesh face.
#[derive(Debug, PartialEq, Default, Clone)]
pub struct PolyMeshFace {
    pub v: Vec<VertIdx>,
    pub vn: Vec<VertNIdx>,
    pub vt: Vec<VertTIdx>,
}

/// Represents a single OBJ file, as well as materials for that OBJ file.
#[derive(Default, Clone)]
pub struct Obj {
    pub objects: Vec<ObjObject>,
    // TODO add groups? What's the difference between groups and objects?
    pub mtls: Vec<(String, MTL)>,

    pub mtllibs: Vec<String>,

    pub(crate) input_file: String,
}

// TODO need to implement a way to fuse a bunch of MTL files into a single super Material.
#[derive(Debug, Clone, Default)]
pub struct ObjObject {
    pub v: Vec<Vec3>,
    pub vt: Vec<Vec2>,
    pub vn: Vec<Vec3>,

    pub f: Vec<PolyMeshFace>,

    /// # faces -> Material
    pub mat: Vec<(Range<FaceIdx>, MatIdx)>,
}

impl From<MeshFace> for PolyMeshFace {
    fn from(mf: MeshFace) -> Self {
        Self::from_mesh_face(mf)
    }
}

impl PolyMeshFace {
    fn from_mesh_face(mf: MeshFace) -> Self {
        Self {
            v: mf.v.to_vec(),
            vn: mf.vn.map(|vn| vn.to_vec()).unwrap_or_default(),
            vt: mf.vt.map(|vt| vt.to_vec()).unwrap_or_default(),
        }
    }
    // Does a simple fan of faces
    pub fn to_mesh_faces(&self) -> impl Iterator<Item = MeshFace> + '_ {
        let root_v = self.v[0];
        let v_iter = self.v[1..].array_windows::<2>();
        let root_vn = self.vn.first().copied();
        let mut vn_iter = self.vn.array_windows::<2>().skip(1);
        let root_vt = self.vt.first().copied();
        let mut vt_iter = self.vt.array_windows::<2>().skip(1);
        v_iter.map(move |&[vi0, vi1]| MeshFace {
            v: [root_v, vi0, vi1],
            vn: root_vn.and_then(|root_vn| {
                let &[vn1, vn2] = vn_iter.next()?;
                Some([root_vn, vn1, vn2])
            }),
            vt: root_vt.and_then(|root_vt| {
                let &[vt1, vt2] = vt_iter.next()?;
                Some([root_vt, vt1, vt2])
            }),
        })
    }
    /// Mutable iterator over all vertex indices
    pub fn v_mut(&mut self) -> impl Iterator<Item = &mut usize> + '_ {
        self.v.iter_mut()
    }
}

impl ObjObject {
    /// Checks if this ObjObject contains any data
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.v.is_empty() && self.vt.is_empty() && self.vn.is_empty() && self.f.is_empty()
    }
    /// Returns the set of poly mesh faces in this corresponding to a specific material index.
    #[inline]
    pub fn mat_faces(&self) -> impl Iterator<Item = (&[PolyMeshFace], MatIdx)> + '_ {
        self.mat.iter().map(|(fs, mi)| (&self.f[fs.clone()], *mi))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MTL {
    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub ke: Vec3,

    /// diffuse map
    pub map_kd: Option<DynamicImage>,
    pub map_kd_path: String,
    /// specular map
    pub map_ks: Option<DynamicImage>,
    pub map_ks_path: String,
    /// ambient map
    pub map_ka: Option<DynamicImage>,
    pub map_ka_path: String,
    /// emissive map
    pub map_ke: Option<DynamicImage>,
    pub map_ke_path: String,
    /// Normal Map
    pub bump_normal: Option<DynamicImage>,
    pub bump_normal_path: String,

    /// Bump/Height Map
    pub disp: Option<DynamicImage>,

    /// Ambient Occlusion Map
    pub map_ao: Option<DynamicImage>,

    /// Read but don't do anything yet
    pub ns: f32,
    pub ni: f32,
    pub d: f32,

    pub illum: u8,
}

impl Default for MTL {
    fn default() -> Self {
        Self {
            ka: Default::default(),
            kd: Default::default(),
            ks: Default::default(),
            ke: Default::default(),

            map_kd: None,
            map_ks: None,
            map_ka: None,
            map_ke: None,
            bump_normal: None,
            disp: None,
            map_ao: None,

            bump_normal_path: String::new(),
            map_ka_path: String::new(),
            map_ke_path: String::new(),
            map_kd_path: String::new(),
            map_ks_path: String::new(),

            ns: 0.,
            ni: 0.,
            // IMPORTANT set this to 1 otherwise it will be transparent.
            d: 1.,
            illum: 0,
        }
    }
}

impl MTL {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
    pub fn diffuse_image(&self) -> DynamicImage {
        if let Some(diffuse) = self.map_kd.as_ref() {
            return diffuse.clone();
        }
        let mut out = DynamicImage::new_rgb32f(1, 1).into_rgb32f();
        out.put_pixel(0, 0, image::Rgb(self.kd.map(|f| f as f32)));
        out.into()
    }
}

fn parse_face(
    f0: &str,
    f1: &str,
    f2: &str,
    num_v: usize,
    num_vt: usize,
    num_vn: usize,
) -> MeshFace {
    let pusize = |v: &str| match v.parse::<i64>() {
        Ok(x) if x < 0 => {
            let x = (-x) as usize;
            assert!(x <= num_v, "{x} {num_v}");
            num_v - x + 1
        }
        Ok(x) => x as usize,
        Err(e) => panic!("Invalid face {v}: {e:?}"),
    };
    let popt = |v: Option<&str>, lim: usize| match v?.parse::<i64>() {
        Ok(x) if x < 0 => {
            let x = (-x) as usize;
            assert!(x <= lim, "{x} {lim}");
            Some(lim - x + 1)
        }
        Ok(x) => Some(x as usize),
        Err(e) if e.kind() == &std::num::IntErrorKind::Empty => None,
        Err(e) => panic!("Invalid face {}: {e:?}", v.unwrap()),
    };

    let split_slash = |v: &str| -> (usize, Option<usize>, Option<usize>) {
        let mut iter = v.split('/');
        match [iter.next(), iter.next(), iter.next()] {
            [None, _, _] => panic!("Missing vertex index in {v}"),
            [Some(a), b, c] => (pusize(a), popt(b, num_vt), popt(c, num_vn)),
        }
    };
    let (v0, vt0, vn0) = split_slash(f0);
    let (v1, vt1, vn1) = split_slash(f1);
    let (v2, vt2, vn2) = split_slash(f2);
    let v = [v0, v1, v2].map(|v| v - 1);
    let vt = vt0.and_then(|vt0| Some([vt0 - 1, vt1? - 1, vt2? - 1]));
    let vn = vn0.and_then(|vn0| Some([vn0 - 1, vn1? - 1, vn2? - 1]));
    MeshFace { v, vt, vn }
}

fn parse_poly_face(fs: &[&str], num_v: usize, num_vt: usize, num_vn: usize) -> PolyMeshFace {
    let pusize = |v: &str| match v.parse::<i64>() {
        Ok(x) if x < 0 => {
            let x = (-x) as usize;
            assert!(x <= num_v, "{x} {num_v}");
            num_v - x + 1
        }
        Ok(x) => x as usize,
        Err(e) => panic!("Invalid face {v}: {e:?}"),
    };
    let popt = |v: Option<&str>, lim: usize| match v?.parse::<i64>() {
        Ok(x) if x < 0 => {
            let x = (-x) as usize;
            assert!(x <= lim, "{x} {lim}");
            Some(lim - x + 1)
        }
        Ok(x) => Some(x as usize),
        Err(e) if e.kind() == &std::num::IntErrorKind::Empty => None,
        Err(e) => panic!("Invalid face {}: {e:?}", v.unwrap()),
    };

    let split_slash = |v: &str| -> (usize, Option<usize>, Option<usize>) {
        let mut iter = v.split('/');
        match [iter.next(), iter.next(), iter.next()] {
            [None, _, _] => panic!("Missing vertex index in {v}"),
            [Some(a), b, c] => (pusize(a), popt(b, num_vt), popt(c, num_vn)),
        }
    };
    let (v, (vt, vn)): (Vec<_>, (Vec<_>, Vec<_>)) = fs
        .iter()
        .map(|f| split_slash(f))
        .map(|(v, vt, vn)| (v - 1, (vt.map(|vt| vt - 1), vn.map(|vn| vn - 1))))
        .unzip();
    let vt: Vec<_> = vt.into_iter().flatten().collect();
    let vn: Vec<_> = vn.into_iter().flatten().collect();
    PolyMeshFace { v, vt, vn }
}
/// Parses a file specified by path `p`.
/// If split_by_object, will return different objects for each group.
pub fn parse(p: impl AsRef<Path>, split_by_object: bool, split_by_group: bool) -> io::Result<Obj> {
    let p = p.as_ref();
    let f = File::open(p)?;
    let buf_read = BufReader::new(f);
    let mut obj = Obj::default();
    obj.input_file = p.to_str().unwrap().into();
    let mut curr_obj = ObjObject::default();
    let mut curr_mtl = None;
    let mut mtl_start_face = 0;
    let mut curr_mtl_name = String::from("");
    let mut logged_no_mtl = false;

    let pf = |v: &str| v.parse::<F>().unwrap();

    for (i, l) in buf_read.lines().enumerate() {
        let l = l?;
        let mut iter = l.split_whitespace();
        let Some(kind) = iter.next() else { continue };
        match kind {
            // comment
            ht if ht.starts_with('#') => continue,
            "v" => match [iter.next(), iter.next(), iter.next()] {
                [None, _, _] | [_, None, _] | [_, _, None] => panic!("Unsupported `v` {i}: {l}"),
                [Some(a), Some(b), Some(c)] => {
                    curr_obj.v.push([pf(a), pf(b), pf(c)]);
                }
            },
            "vt" => match [iter.next(), iter.next(), iter.next()] {
                [None, _, _] | [_, None, _] => panic!("Unsupported `vt` {i}: {l}"),
                [Some(a), Some(b), _] => {
                    curr_obj.vt.push([pf(a), pf(b)]);
                }
            },
            "vn" => match [iter.next(), iter.next(), iter.next()] {
                [None, _, _] | [_, None, _] | [_, _, None] => panic!("Unsupported `vn` {i}: {l}"),
                [Some(a), Some(b), Some(c)] => {
                    curr_obj.vn.push([pf(a), pf(b), pf(c)]);
                }
            },
            "f" => match [iter.next(), iter.next(), iter.next(), iter.next()] {
                [Some(a), Some(b), Some(c), Some(d)] => {
                    let mut all_verts = vec![a, b, c, d];
                    all_verts.extend(iter);
                    let co = &curr_obj;
                    curr_obj.f.push(parse_poly_face(
                        &all_verts,
                        co.v.len(),
                        co.vt.len(),
                        co.vn.len(),
                    ));
                }
                [None, _, _, _] | [_, None, _, _] | [_, _, None, _] => {
                    panic!("Unsupported `f` format {l}")
                }
                [Some(a), Some(b), Some(c), None] => {
                    if a == b || b == c || a == c {
                        eprintln!("Face contains multiple of the same vertex, ignoring");
                        continue;
                    }
                    let co = &curr_obj;
                    let f = parse_face(a, b, c, co.v.len(), co.vt.len(), co.vn.len()).into();
                    curr_obj.f.push(f);
                }
            },
            "g" if !split_by_group => continue,
            "g" => {
                if !curr_obj.is_empty() {
                    obj.objects.push(curr_obj.clone());
                    curr_obj.f.clear();
                }
            }
            "o" if !split_by_object => continue,
            "o" => {
                if !curr_obj.is_empty() {
                    obj.objects.push(curr_obj.clone());
                    curr_obj.f.clear();
                }
            }
            // TODO not sure what to do for smoothing groups
            "s" => {}
            "mtllib" => {
                let Some(mtl_file) = iter.remainder() else {
                    panic!("Missing mtl file in {l}")
                };
                // Try a bunch of different attempts
                match parse_mtl(mtl_file) {
                    Ok(mtls) => {
                        obj.mtllibs.push(String::from(mtl_file));
                        obj.mtls.extend(mtls);
                        continue;
                    }
                    Err(_e) => {}
                };
                let appended = p.with_file_name(mtl_file);
                match parse_mtl(&appended) {
                    Ok(mtls) => {
                        obj.mtllibs.push(String::from(appended.to_str().unwrap()));
                        obj.mtls.extend(mtls);
                        continue;
                    }
                    Err(_e) => {}
                };
                if let Some(file) = PathBuf::from(mtl_file).file_name() {
                    match parse_mtl(file) {
                        Ok(mtls) => {
                            obj.mtllibs.push(String::from(file.to_str().unwrap()));
                            obj.mtls.extend(mtls);
                            continue;
                        }
                        Err(_e) => {}
                    }
                }
            }
            "usemtl" if obj.mtls.is_empty() => {
                if !logged_no_mtl {
                    logged_no_mtl = true;
                    eprintln!("[WARN]: OBJ with materials, but no mtls found");
                }
            }
            "usemtl" => {
                // This should also cause the object to split into a different object.
                let Some(mtl_name) = iter.remainder() else {
                    panic!("Missing mtl name in {l}")
                };
                // if the previous mtl is the same, continue and don't do anything
                if curr_mtl_name == mtl_name {
                    continue;
                }

                curr_mtl_name = mtl_name.into();
                let Some(mtl_idx) = obj.mtls.iter().position(|mtl| mtl.0 == mtl_name) else {
                    eprintln!(
                        "[WARN]: Could not find mtl {mtl_name}, have {:?}",
                        obj.mtls.iter().map(|(n, _)| n).collect::<Vec<_>>()
                    );
                    continue;
                };
                if let Some(mtl) = curr_mtl {
                    curr_obj.mat.push((mtl_start_face..curr_obj.f.len(), mtl));
                }
                mtl_start_face = curr_obj.f.len();
                curr_mtl = Some(mtl_idx);
            }
            "l" => {
                static mut DID_WARN_LINES: bool = false;
                unsafe {
                    if !DID_WARN_LINES {
                        eprintln!("Line elements not currently handled: {l}");
                        DID_WARN_LINES = true;
                    }
                }
            }
            // TODO
            k => eprintln!("[ERROR]: Unknown line in OBJ {l} with {k:?}"),
        };
    }
    if let Some(idx) = curr_mtl {
        curr_obj.mat.push((mtl_start_face..curr_obj.f.len(), idx));
    }

    if !curr_obj.is_empty() {
        obj.objects.push(curr_obj);
    }
    Ok(obj)
}

pub fn parse_mtl(p: impl AsRef<Path>) -> io::Result<Vec<(String, MTL)>> {
    let f = File::open(p.as_ref())?;
    let buf_read = BufReader::new(f);
    let mut curr_mtl = MTL::default();
    let mut curr_name = String::new();

    let pf = |v: &str| v.parse::<F>().unwrap();
    let pu8 = |v: &str| v.parse::<u8>().unwrap();
    let mut out = vec![];
    for l in buf_read.lines() {
        let l = l?;
        let mut iter = l.split_whitespace();
        let Some(kind) = iter.next() else { continue };
        let kind = kind.to_lowercase();
        match kind.as_str() {
            ht if ht.starts_with('#') => continue,
            "kd" | "ks" | "ka" | "ke" | "tf" => match [iter.next(), iter.next(), iter.next()] {
                [None, _, _] | [_, None, _] | [_, _, None] => panic!("Unsupported {kind} {l}"),
                [Some(r), Some(g), Some(b)] => {
                    *match kind.as_str() {
                        "kd" => &mut curr_mtl.kd,
                        "ks" => &mut curr_mtl.ks,
                        "ka" => &mut curr_mtl.ka,
                        "ke" => &mut curr_mtl.ke,
                        "tf" => continue,
                        _ => unreachable!(),
                    } = [pf(r), pf(g), pf(b)];
                }
            },
            "map_kd" | "map_ka" | "map_ke" | "map_ks" | "disp" | "bump_normal" | "map_normal"
            | "bump" | "map_bump" | "map_ao" | "map_ns" | "refl" | "map_d" => {
                let f = match [iter.next(), iter.next(), iter.next()] {
                    [Some(f), None, _] => f,
                    [Some(_), Some(_), Some(f)] => f,
                    [Some(_), Some(_), None] => panic!("Unknown format for line {l}"),
                    [None, _, _] => panic!("Missing file in {l}"),
                };
                assert_eq!(iter.remainder(), None, "Unknown format for line {l}");
                let file_path = PathBuf::from(f);

                let mtl_path = if file_path.is_absolute() {
                    p.as_ref().with_file_name(file_path.file_name().unwrap())
                } else {
                    p.as_ref().with_file_name(file_path)
                };

                let windows_file = if let Some(reverse_path) = f.rsplit('\\').next() {
                    p.as_ref().with_file_name(reverse_path)
                } else {
                    // just a dummy
                    mtl_path.clone()
                };

                let choices = [
                    mtl_path.clone(),
                    PathBuf::from(f),
                    // TODO do these lazily?

                    // These really shouldn't be here
                    // But because of errors don't do it
                    mtl_path.clone().with_extension("png"),
                    mtl_path.clone().with_extension("jpg"),
                    mtl_path.clone().with_extension("jpeg"),
                    windows_file.clone().with_extension("png"),
                    windows_file.clone().with_extension("jpg"),
                    windows_file.clone().with_extension("jpeg"),
                ];
                let img_c = choices
                    .into_iter()
                    .find_map(|c| Some((image::open(&c).ok()?, c)));
                let Some((img, c)) = img_c else {
                    // TODO retry with appending mtl path with texture.
                    eprintln!("Failed to load image {f:?}");
                    continue;
                };

                let (img_dst, path_dst) = match kind.as_str() {
                    "map_kd" => (&mut curr_mtl.map_kd, Some(&mut curr_mtl.map_kd_path)),
                    "map_ks" => (&mut curr_mtl.map_ks, Some(&mut curr_mtl.map_ks_path)),
                    "map_ka" => (&mut curr_mtl.map_ka, Some(&mut curr_mtl.map_ka_path)),
                    "map_ke" => (&mut curr_mtl.map_ke, Some(&mut curr_mtl.map_ke_path)),
                    "disp" => (&mut curr_mtl.disp, None),
                    "map_ao" => (&mut curr_mtl.map_ao, None),
                    "bump" | "map_bump" | "map_normal" | "bump_normal" => (
                        &mut curr_mtl.bump_normal,
                        Some(&mut curr_mtl.bump_normal_path),
                    ),
                    // TODO need to implement these
                    "map_ns" => continue,
                    "map_d" => continue,
                    "refl" => continue,
                    _ => unreachable!(),
                };
                *img_dst = Some(img);
                if let Some(path_dst) = path_dst {
                    *path_dst = c.to_str().unwrap().into();
                }
            }
            "newmtl" => {
                let old = std::mem::take(&mut curr_mtl);
                let new_name = iter.next().expect("missing name");
                let old_name = std::mem::replace(&mut curr_name, new_name.to_string());
                if !old.is_empty() {
                    out.push((old_name, old));
                }
            }
            "ns" | "ni" | "d" | "tr" => {
                let Some(v) = iter.next() else {
                    panic!("Missing value in {l}")
                };
                *match kind.as_str() {
                    "ni" => &mut curr_mtl.ni,
                    "ns" => &mut curr_mtl.ns,
                    "d" => &mut curr_mtl.d,
                    "tr" => {
                        curr_mtl.d = (1. - pf(v)) as f32;
                        continue;
                    }
                    _ => unreachable!(),
                } = pf(v) as f32;
            }
            "illum" => {
                let Some(n) = iter.next() else {
                    panic!("Missing value in {l}")
                };
                curr_mtl.illum = pu8(n);
            }
            // This is a random tag
            k => println!("[ERROR]: Unknown line in MTL {k:?}"),
        }
    }
    if !curr_mtl.is_empty() {
        out.push((curr_name, curr_mtl));
    }
    Ok(out)
}

impl MTL {
    pub fn write(&self, mut dst: impl Write, name_prefix: impl AsRef<OsStr>) -> io::Result<()> {
        let name_prefix = name_prefix.as_ref().to_str().unwrap();
        dst.write_all(b"# generated by pars3d\n")?;
        let mtl_name = Path::new(&name_prefix)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        writeln!(dst, "newmtl {mtl_name}")?;

        macro_rules! write_image {
            ($img : expr, $name_suffix : expr, $mtl_img_string: expr $(,)?) => {
                if let Some(map) = $img {
                    let name = format!("{name_prefix}_{}.png", $name_suffix);
                    map.save(&name)
                        .unwrap_or_else(|_| panic!("Failed to save {name}"));
                    let name = Path::new(&name).file_name().unwrap().to_str().unwrap();
                    writeln!(dst, "{} {name}", $mtl_img_string)?;
                }
            };
        }

        write_image!(&self.map_kd, "kd", "map_Kd",);
        writeln!(dst, "Kd {} {} {}", self.kd[0], self.kd[1], self.kd[2])?;

        write_image!(&self.map_ks, "ks", "map_Ks",);
        writeln!(dst, "Ks {} {} {}", self.ks[0], self.ks[1], self.ks[2])?;

        write_image!(&self.map_ka, "ka", "map_Ka",);
        writeln!(dst, "Ka {} {} {}", self.ka[0], self.ka[1], self.ka[2])?;

        write_image!(&self.map_ke, "ke", "map_Ke",);
        writeln!(dst, "Ke {} {} {}", self.ke[0], self.ke[1], self.ke[2])?;

        write_image!(&self.bump_normal, "n", "bump",);

        write_image!(&self.disp, "disp", "disp",);

        write_image!(&self.map_ao, "ao", "map_ao",);

        writeln!(dst, "Ns {}", self.ns)?;
        writeln!(dst, "Ni {}", self.ni)?;
        writeln!(dst, "d {}", self.d)?;
        writeln!(dst, "illum {}", self.illum)?;

        Ok(())
    }
}

impl Obj {
    /// Writes this set of obj objects and mtls to writers.
    pub fn write(&self, mut dst: impl Write, dst_dir: &str) -> io::Result<()> {
        for mtllib in &self.mtllibs {
            writeln!(dst, "mtllib {mtllib}")?;
        }
        for object in &self.objects {
            object.write(&mut dst, &self.mtls)?;
        }
        // not sure if there is any way to figure out which mtl came from which mtllib
        // unless it's tracked from the beginning
        for mtllib in &self.mtllibs {
            let mtl_path = Path::new(dst_dir).join(mtllib);
            let f = File::create(mtl_path)?;
            let mut f = BufWriter::new(f);
            for (name, mtl) in self.mtls.iter() {
                mtl.write(&mut f, Path::new(dst_dir).join(name))?;
            }
        }
        Ok(())
    }
}

impl ObjObject {
    /// Writes this obj object out to a writer
    pub fn write(&self, mut dst: impl Write, mtl_names: &[(String, MTL)]) -> io::Result<()> {
        for v in &self.v {
            // always write out as a float.
            let [x, y, z] = v;
            writeln!(dst, "v {x} {y} {z}")?;
        }

        for vt in &self.vt {
            let [u, v] = vt;
            writeln!(dst, "vt {u} {v}")?;
        }

        for vn in &self.vn {
            let [x, y, z] = vn;
            writeln!(dst, "vn {x} {y} {z}")?;
        }

        // TODO maybe figure out whether this is correct or not?
        dst.write_all(b"s 1\n")?;
        dst.write_all(b"g mesh_1\n")?;
        let mut curr_mat = None;
        for (r, mat_idx) in &self.mat {
            if !r.contains(&0) {
                continue;
            }
            assert_eq!(curr_mat, None);
            let first_mtl = mtl_names
                .get(*mat_idx)
                .map(|m| m.0.clone())
                .unwrap_or_else(|| String::from("default_mat"));
            curr_mat = Some(*mat_idx);
            writeln!(dst, "usemtl {first_mtl}")?;
        }

        for (fi, f) in self.f.iter().enumerate() {
            if f.v.is_empty() {
                continue;
            }
            assert_ne!(f.v.len(), 1);
            assert_ne!(f.v.len(), 2);
            if let Some(&(_, mat_idx)) = self.mat.iter().find(|(r, _)| r.contains(&fi)) {
                if Some(mat_idx) != curr_mat {
                    let mtl_name = mtl_names
                        .get(mat_idx)
                        .map(|m| m.0.clone())
                        .unwrap_or_else(|| format!("default_mat{mat_idx}"));
                    writeln!(dst, "usemtl {mtl_name}")?;
                    curr_mat = Some(mat_idx);
                }
            }

            dst.write_all(b"f ")?;
            let mut vn = f.vn.iter().map(|vn| vn + 1);
            let mut vt = f.vt.iter().map(|vt| vt + 1);
            for v in &f.v[0..f.v.len() - 1] {
                let v = v + 1;
                match (vn.next(), vt.next()) {
                    (None, None) => write!(dst, "{v} "),
                    (Some(vn), None) => write!(dst, "{v}//{vn} "),
                    (None, Some(vt)) => write!(dst, "{v}/{vt} "),
                    (Some(vn), Some(vt)) => write!(dst, "{v}/{vt}/{vn} "),
                }?;
            }
            // the last one is separate to remove the trailing space.
            let v = f.v.last().unwrap();
            let v = v + 1;
            match (vn.next(), vt.next()) {
                (None, None) => writeln!(dst, "{v}"),
                (Some(vn), None) => writeln!(dst, "{v}//{vn}"),
                (None, Some(vt)) => writeln!(dst, "{v}/{vt}"),
                (Some(vn), Some(vt)) => writeln!(dst, "{v}/{vt}/{vn}"),
            }?;
        }
        Ok(())
    }
}

/// How to output an MTL
#[derive(Debug, Clone, PartialEq)]
pub enum OutputKind {
    /// Reuse the original file if it exists, with an absolute file path
    ReuseAbsolute,
    /// Reuse the original file if it exists, with relative file path to the one provided.
    ReuseRelative(String),

    /// Generate a new mtl at this location. Cannot be empty.
    New(String),

    /// Do not write out any MTL file.
    None,
}

/// Writes all materials to a single MTL file (with corresponding images)
/// Returns Ok(Some(path to mtl)) on success AND if there is an mtl file.
fn write_mtls(
    s: &super::mesh::Scene,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(super::mesh::TextureKind, &str) -> OutputKind,
) -> io::Result<Option<String>> {
    let mtl_output_kind = if s.mtllibs.is_empty() {
        mtl_file("")
    } else {
        mtl_file(&s.mtllibs[0])
    };

    use std::fs::exists;
    let mtl_path = match mtl_output_kind {
        OutputKind::None => return Ok(None),
        OutputKind::ReuseAbsolute
            if s.mtllibs.len() == 1 && exists(&s.mtllibs[0]).unwrap_or(false) =>
        {
            let dst_path: &Path = s.mtllibs[0].as_ref();
            if dst_path.is_absolute() {
                return Ok(Some(String::from(&s.mtllibs[0])));
            }
            let out_path = std::path::absolute(&dst_path)?;

            return Ok(Some(String::from(out_path.to_str().unwrap())));
        }
        OutputKind::ReuseRelative(out_file)
            if s.mtllibs.len() == 1 && exists(&s.mtllibs[0]).unwrap_or(false) =>
        {
            let dst_path: &Path = s.mtllibs[0].as_ref();
            // if original path was absolute don't switch to relative.
            if dst_path.is_absolute() {
                return Ok(Some(String::from(&s.mtllibs[0])));
            }

            let out_path: &Path = out_file.as_ref();
            assert!(out_path.is_file());
            let out_path = std::path::absolute(out_path)?;

            let dst_path = std::path::absolute(dst_path)?;

            // Path from out file *to* original obj file *to* original mtl file
            let out_to_mtl = {
                let mut curr = out_path.parent();
                let mut num_parents = 0;
                while let Some(c) = curr
                    && !dst_path.starts_with(c)
                {
                    num_parents += 1;
                    curr = c.parent();
                }
                let prefix = match curr {
                    None => todo!(),
                    Some(c) => dst_path.strip_prefix(c).unwrap(),
                };
                let mut backs = PathBuf::new();
                backs.extend((0..num_parents).map(|_| "../"));
                backs.push(prefix);
                backs
            };

            return Ok(Some(String::from(out_to_mtl.to_str().unwrap())));
        }
        OutputKind::ReuseAbsolute | OutputKind::ReuseRelative(_) => {
            panic!("Unable to reuse input MTL file: {}", s.mtllibs[0])
        }
        OutputKind::New(v) => v,
    };
    assert_ne!(mtl_path, "", "Must not pass empty MTL");

    let mtl_file = File::create(&mtl_path)?;
    let mut mtl_file = BufWriter::new(mtl_file);

    for (mi, mat) in s.materials.iter().enumerate() {
        let mat_name = if mat.name.is_empty() {
            mat.name.clone()
        } else {
            format!("mat_{mi}")
        };
        writeln!(mtl_file, "newmtl {mat_name}")?;
        for tex in &mat.textures {
            macro_rules! save_img {
                ($img: expr) => {{
                    let img = $img;
                    let (save, path) = match img_dsts(tex.kind, &tex.original_path) {
                        OutputKind::None => continue,
                        OutputKind::ReuseAbsolute | OutputKind::ReuseRelative(_)
                            if exists(&tex.original_path).unwrap_or(false) =>
                        {
                            (false, String::from(&tex.original_path))
                        }
                        OutputKind::ReuseAbsolute | OutputKind::ReuseRelative(_) => {
                            (true, format!("{:?}.png", tex.kind))
                        }
                        OutputKind::New(f) => (true, f),
                    };
                    if save {
                        match img.save(&path) {
                            Ok(()) => {}
                            Err(image::ImageError::IoError(err)) => return Err(err),
                            Err(e) => panic!("Failed to save image in OBJ: {e:?}"),
                        }
                    }
                    path
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
                    writeln!(mtl_file, "{} {path}", $img_name)?;
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
                    writeln!(mtl_file, "bump {path}")?;
                }
                // OBJ cannot handle other kinds of textures
                _ => continue,
            }
        }
    }

    Ok(Some(mtl_path))
}

pub fn save_obj(
    s: &super::mesh::Scene,
    mut geom_dst: impl Write,
    // Path to MTL file if any. Original MTL file is also passed.
    mtl_file: impl Fn(&str) -> OutputKind,
    // given texture kind & original path, output new path to write to
    img_dsts: impl Fn(super::mesh::TextureKind, &str) -> OutputKind,
) -> io::Result<()> {
    let has_materials =
        !s.materials.is_empty() && s.materials.iter().any(|m| !m.textures.is_empty());
    let mtl_path = if has_materials {
        write_mtls(s, mtl_file, img_dsts)?
    } else {
        None
    };
    //let mtls = s.mtllibs.into_iter().map()
    writeln!(geom_dst, "# Generated by pars3d")?;
    if let Some(mtl_path) = mtl_path {
        writeln!(geom_dst, "mtllib {mtl_path}")?;
    }

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
        for f in &m.f {
            geom_dst.write_all(b"f ")?;
            let Some((lv, fv)) = f.as_slice().split_last() else {
                continue;
            };
            for &v in fv {
                write!(geom_dst, "{} ", fmt(v + 1))?;
            }
            writeln!(geom_dst, "{}", fmt(lv + 1))?;
        }
    }
    Ok(())
}

#[test]
fn test_load_save_obj() {
    let scene: super::mesh::Scene = parse("garlic.obj", false, false)
        .expect("Failed to parse obj")
        .into();
    let obj_out = File::create("garlic_tmp.obj").unwrap();
    save_obj(
        &scene,
        BufWriter::new(obj_out),
        |_mtl_path| OutputKind::New(String::from("garlic_tmp.mtl")),
        |_tex_kind, _og_path| OutputKind::ReuseAbsolute,
    )
    .expect("Failed to save");
}
