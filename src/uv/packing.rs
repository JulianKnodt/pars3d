use super::conservative_rasterize::conservative_rasterize;
use crate::aabb::AABB;
use crate::grid::Arr2D;
use crate::{F, FaceKind, OrdFloat, sub};

use rustfft::num_complex::Complex;

#[derive(Debug)]
pub struct PackingArgs<'a> {
    //vs: &'a [[F; 3]],
    uvs: &'a [[F; 2]],
    faces: &'a [FaceKind],

    // TODO not sure if this should be included this way
    chart_assns: Vec<u32>,
    num_charts: u32,

    target_width: usize,
    target_height: usize,
}

impl<'a> PackingArgs<'a> {
    pub fn new(uvs: &'a [[F; 2]], faces: &'a [FaceKind], tw: usize, th: usize) -> Self {
        let ff_adj = crate::adjacency::face_face_adj(faces);
        let (chart_assns, num_charts) = ff_adj.connected_components();
        Self {
            uvs,
            faces,
            chart_assns,
            num_charts,
            target_width: tw,
            target_height: th,
        }
    }
}

#[allow(unused)]
pub fn fft_pack(dst: &mut Vec<[F; 2]>, args: &PackingArgs) {
    assert_eq!(args.faces.len(), args.chart_assns.len());
    dst.resize(args.uvs.len(), [0.; 2]);
    // rescale charts so they fit into [-1, 1] (or no rescaling then uniform rescale)
    let mut chart_areas_2d = vec![0.; args.num_charts as usize];
    for (f, &ci) in args.faces.iter().zip(args.chart_assns.iter()) {
        chart_areas_2d[ci as usize] += f.area_2d(args.uvs);
    }

    let total_area_2d = chart_areas_2d.iter().sum::<F>();

    let mut chart_ord = (0..args.num_charts).collect::<Vec<_>>();
    chart_ord.sort_unstable_by_key(|&ci| std::cmp::Reverse(OrdFloat(chart_areas_2d[ci as usize])));

    println!(
        "{:?} {:?}",
        chart_areas_2d[chart_ord[0] as usize], chart_areas_2d[chart_ord[1] as usize]
    );

    let tw = args.target_width;
    let th = args.target_height;

    // current set of charts
    let mut total_grid = Arr2D::<Complex<F>>::from_fn(tw, th, |x, y| {
        if x == 1 || y == 1 {
            return Complex { re: 1., im: 0. };
        }
        Complex::default()
    });
    let mut total_spectral = Arr2D::<Complex<F>>::empty(tw, th);

    let mut chart_buf = Arr2D::<Complex<F>>::empty(tw, th);
    let mut buf_spectral = Arr2D::<Complex<F>>::empty(tw, th);

    let mut valid_regions = Arr2D::<Complex<F>>::empty(tw, th);

    let mut planner = rustfft::FftPlanner::new();

    // insert charts one-at-a-time, largest
    for ci in chart_ord {
        total_grid.fft(&mut planner, &mut total_spectral);
        // TODO here perform multiple rotations
        // rasterize charts
        let mut aabb = AABB::new();
        for (fi, f) in args.faces.iter().enumerate() {
            if args.chart_assns[fi] != ci {
                continue;
            }
            for &vi in f.as_slice() {
                aabb.add_point(args.uvs[vi]);
            }
        }

        // rasterize chart into top-left corner
        chart_buf.fill_with(Complex::default);

        for (fi, f) in args.faces.iter().enumerate() {
            if args.chart_assns[fi] != ci {
                continue;
            }
            // TODO in theory this may be wrong for non-convex polygons
            for t in f.as_triangle_fan() {
                let uv_t = t.map(|vi| sub(args.uvs[vi], aabb.min));
                conservative_rasterize(
                    uv_t,
                    tw,
                    th,
                    &mut chart_buf,
                    |g, ij| *g.get(ij).unwrap_or(&Complex::default()) != Default::default(),
                    |g, ij| g.get_mut(ij).unwrap().re = 1.,
                );
            }
        }

        // find place where chart can fit

        // take fft of chart, elementwise multiply,
        chart_buf.fft(&mut planner, &mut buf_spectral);

        // convolve both
        buf_spectral.elemwise_op_assign(&total_spectral, |&a, &b| a * b);

        // inverse fft
        //buf_spectral.ifft(&mut planner, &mut valid_regions);

        /* TODO tmp */
        let f = std::fs::File::create("tmp.ppm").unwrap();
        use crate::ppm::write as write_ppm;
        use std::io::Write;
        write_ppm(f, 1024, 1024, |i, j| {
            let r = buf_spectral.get([i, j]).unwrap().re;
            let b = (r.clamp(0., 1.) * 255.) as u8;
            [b; 3]
        })
        .unwrap();
        // ---
        todo!();
    }
}

#[test]
fn test_fft_pack() {
    let mut o = crate::load("data/sphere_with_charts.obj").unwrap();
    assert_eq!(o.meshes.len(), 1);
    let m = o.meshes.pop().unwrap();

    let packing_args = PackingArgs::new(&m.uv[0], &m.f, 1024, 1024);
    let mut out_uv = vec![];
    fft_pack(&mut out_uv, &packing_args);

    todo!();
}
