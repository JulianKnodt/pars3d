use pars3d::Mesh;
use pars3d::geom_processing::subdivision::honeycomb_tet;
use pars3d::mesh::Line;

#[test]
fn test_subdiv_tet() -> std::io::Result<()> {
    let tet_v = [
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.],
        [0., -1., 0.],
        [-1., 0., 0.],
    ];

    let tet_f = [
        [0, 1, 2, 3], //
        [1, 2, 3, 4],
        [0, 5, 2, 3],
    ];

    let (new_v, _, edges) = honeycomb_tet(&tet_v, &tet_f, 0.1);

    /*
    let mut cnts = vec![0; new_v.len()];
    for [vi0, vi1] in edges.iter().copied() {
      cnts[vi0] += 1;
      cnts[vi1] += 1;
    }
    let max = cnts.into_iter().max().unwrap();
    assert!(max == 4);
    */

    let mut mesh = Mesh::new_geometry(new_v, vec![]);
    let new_lines = edges
        .into_iter()
        .map(|[e0, e1]| Line::new_from_endpoints(e0, e1));
    mesh.l = new_lines.collect();

    pars3d::save("tet_subdiv_tmp.obj", &mesh.into_scene())
}
