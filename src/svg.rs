use crate::{FaceKind, F};
use svg::{
    node::element::{path::Data, Path},
    Document,
};

/// Save a 2D SVG of a UV map
pub fn save_uv(
    dst: impl AsRef<std::path::Path>,
    uvs: &[[F; 2]],
    // TODO figure out how to add colors to this
    faces: &[FaceKind],
) -> std::io::Result<()> {
    let mut doc = Document::new().set("viewBox", (0., 0., 1., 1.));

    for f in faces {
        if f.is_empty() {
            continue;
        }
        let s = f.as_slice();
        let [u0, v0] = uvs[s[0]];
        let mut data = Data::new().move_to((u0, v0));
        for &vi in &s[1..] {
            let [u, v] = uvs[vi];
            data = data.line_to((u, v));
        }
        data = data.line_to((u0, v0)).close();

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 3)
            .set("d", data);

        doc = doc.add(path);
    }

    svg::save(dst, &doc)
}
