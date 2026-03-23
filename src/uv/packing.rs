use super::conservative_rasterize::conservative_rasterize;
use crate::aabb::AABB;
use crate::grid::Arr2D;
use crate::{F, FaceKind, OrdFloat, sub};

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
pub fn pack(dst: &mut Vec<[F; 2]>, args: &PackingArgs) {
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
    let mut total_grid = Arr2D::<bool>::from_fn(tw, th, |x, y| {
        x == 0 || y == 0 || x == tw - 1 || y == th - 1
    });

    let mut chart_buf = Arr2D::<bool>::empty(tw, th);

    let mut valid_regions = Arr2D::<bool>::empty(tw, th);

    // insert charts one-at-a-time, largest-first
    for ci in chart_ord {
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
        chart_buf.fill(false);

        for (fi, f) in args.faces.iter().enumerate() {
            if args.chart_assns[fi] != ci {
                continue;
            }
            println!("{fi}");
            // TODO in theory this may be wrong for non-convex polygons
            for t in f.as_triangle_fan() {
                let uv_t = t.map(|vi| sub(args.uvs[vi], aabb.min));
                conservative_rasterize(
                    uv_t,
                    tw,
                    th,
                    &mut chart_buf,
                    |g, ij| *g.get(ij).unwrap_or(&false),
                    |g, ij| *g.get_mut(ij).unwrap() = true,
                );
                // TODO add padding here?
            }
        }

        println!("Convolving");
        total_grid.convolve(&chart_buf, &mut valid_regions);

        /* TODO tmp */
        let f = std::fs::File::create("tmp.ppm").unwrap();
        use crate::ppm::write as write_ppm;
        use std::io::Write;
        write_ppm(f, 256, 256, |i, j| {
            let r = *valid_regions.get([i, j]).unwrap();
            let b = if r { 255 } else { 0 };
            [b; 3]
        })
        .unwrap();
        // ---
        todo!();
    }
}

#[test]
fn test_basic_pack() {
    let mut o = crate::load("data/sphere_with_charts.obj").unwrap();
    assert_eq!(o.meshes.len(), 1);
    let m = o.meshes.pop().unwrap();

    let packing_args = PackingArgs::new(&m.uv[0], &m.f, 256, 256);
    let mut out_uv = vec![];
    pack(&mut out_uv, &packing_args);

    todo!();
}
