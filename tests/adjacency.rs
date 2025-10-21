use pars3d::adjacency::Winding;
use pars3d::{FaceKind, Mesh};

#[test]
fn test_vert_face_one_ring() {
    let faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        //[0,6,1],
    ]
    .into_iter()
    .map(FaceKind::Tri)
    .collect();

    let m = Mesh::new_geometry(vec![[0.; 3]; 7], faces);
    let adj = m.vertex_face_adj();
    let mut w = Winding::new();

    adj.vertex_face_one_ring_ord(0, |fi| m.f[fi].as_slice(), &mut w);
    assert_eq!(w.iter().count(), 6);
    assert_eq!(w.num_breaks(), 1);
}
