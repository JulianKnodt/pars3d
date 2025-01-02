use crate::{add, normalize, rotate_on_axis, F};
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// A material from the MERL dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct MERL {
    data: Vec<f64>,
}
pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<MERL> {
    read(std::fs::File::open(p)?)
}

pub fn read(reader: impl Read) -> io::Result<MERL> {
    buf_read(BufReader::new(reader))
}

const BRDF_SAMPLING_RES_THETA_H: usize = 90;
const BRDF_SAMPLING_RES_THETA_D: usize = 90;
const BRDF_SAMPLING_RES_PHI_D: usize = 360;

const RED_SCALE: f64 = 1.0 / 1500.0;
const GREEN_SCALE: f64 = 1.15 / 1500.0;
const BLUE_SCALE: f64 = 1.66 / 1500.0;

pub fn buf_read(mut reader: impl BufRead) -> io::Result<MERL> {
    let mut d0 = [0u8; 4];
    let mut d1 = [0u8; 4];
    let mut d2 = [0u8; 4];
    reader.read_exact(&mut d0)?;
    reader.read_exact(&mut d1)?;
    reader.read_exact(&mut d2)?;
    let d0 = i32::from_le_bytes(d0) as usize;
    let d1 = i32::from_le_bytes(d1) as usize;
    let d2 = i32::from_le_bytes(d2) as usize;
    assert_eq!(BRDF_SAMPLING_RES_THETA_H, d0);
    assert_eq!(BRDF_SAMPLING_RES_THETA_D, d1);
    assert_eq!(BRDF_SAMPLING_RES_PHI_D / 2, d2);

    let data: io::Result<Vec<f64>> = reader
        .bytes()
        .array_chunks::<8>()
        .map(|a| a.try_map(|v| v))
        .map(|a| Ok(f64::from_le_bytes(a?)))
        .collect();
    let data = data?;

    Ok(MERL { data })
}

fn theta_phi_to_vec(theta: F, phi: F) -> [F; 3] {
    let z = theta.cos();
    let p = theta.sin();
    normalize([phi.cos() * p, phi.sin() * p, z])
}

const HALF_PI: F = std::f64::consts::FRAC_PI_2 as F;
const PI: F = std::f64::consts::PI as F;
fn theta_half_index(theta_half: F) -> usize {
    if theta_half <= 0.0 {
        return 0;
    }
    let theta_half_deg = (theta_half / HALF_PI) * BRDF_SAMPLING_RES_THETA_H as F;
    let temp = theta_half_deg * BRDF_SAMPLING_RES_THETA_H as F;
    let temp = temp.sqrt().max(0.) as usize;
    temp.min(BRDF_SAMPLING_RES_THETA_H - 1)
}

fn theta_diff_index(theta_diff: F) -> usize {
    if theta_diff < 0. {
        return 0;
    }
    let tmp = (theta_diff / HALF_PI * BRDF_SAMPLING_RES_THETA_D as F) as usize;
    tmp.min(BRDF_SAMPLING_RES_THETA_D - 1)
}

// Lookup phi_diff index
fn phi_diff_index(phi_diff: F) -> usize {
    // Because of reciprocity, the BRDF is unchanged under
    // phi_diff -> phi_diff + M_PI
    if phi_diff < 0.0 {
        return phi_diff_index(phi_diff + PI);
    }

    // In: phi_diff in [0 .. pi]
    // Out: tmp in [0 .. 179]
    let tmp = (phi_diff / HALF_PI * BRDF_SAMPLING_RES_PHI_D as F) as usize;
    tmp.min(BRDF_SAMPLING_RES_PHI_D / 2 - 1)
}

/// Converts XYZ to R, θ (theta), φ (phi)
pub fn xyz_to_spherical([x, y, z]: [F; 3]) -> [F; 3] {
    let r = (x * x + y * y + z * z).sqrt();
    let phi = (z / r).acos();
    let theta = y.atan2(x);
    [r, theta, phi]
}

impl MERL {
    pub fn eval_dirs(&self, l: [F; 3], v: [F; 3]) -> [f64; 3] {
        let [_, theta_in, phi_in] = xyz_to_spherical(l);
        let [_, theta_out, phi_out] = xyz_to_spherical(v);
        self.eval(theta_in, phi_in, theta_out, phi_out)
    }
    pub fn eval(&self, theta_in: F, phi_in: F, theta_out: F, phi_out: F) -> [f64; 3] {
        let v_in = theta_phi_to_vec(theta_in, phi_in);
        let v_out = theta_phi_to_vec(theta_out, phi_out);
        let h = normalize(add(v_in, v_out));

        let theta_half = h[2].acos();
        let phi_half = h[1].atan2(h[0]);

        let normal = [0., 0., 1.];
        let binormal = [0., 1., 0.];

        let tmp = rotate_on_axis(v_in, normal, -phi_half.sin(), phi_half.cos());
        let diff = rotate_on_axis(tmp, binormal, -theta_half.sin(), theta_half.cos());

        let theta_diff = diff[2].acos();
        let phi_diff = diff[1].atan2(diff[0]);

        self.eval_raw(theta_diff, phi_diff, theta_half)

    }
    /// Evaluate this material using an explicit rusinkiewicz parameterization.
    pub fn eval_raw(&self, theta_diff: F, phi_diff: F, theta_half: F) -> [f64; 3] {
        let ind = phi_diff_index(phi_diff)
            + theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2
            + theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2
                * BRDF_SAMPLING_RES_THETA_D;

        let brdf = &self.data;
        let r = brdf[ind] * RED_SCALE;
        let g_ind = ind
            + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2;
        let g = brdf[g_ind] * GREEN_SCALE;
        let b_ind =
            ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D;
        let b = brdf[b_ind] * BLUE_SCALE;
        [r, g, b]
    }
}

#[test]
pub fn test_load_merl() {
    let yellow_plastic = read_from_file("yellow-plastic.binary").unwrap();
    for i in 0..90 {
        for j in 0..90 {
            yellow_plastic.eval(
                (i as F).to_radians(),
                (j as F).to_radians(),
                (45. as F).to_radians(),
                (45. as F).to_radians(),
            );
        }
    }
}
