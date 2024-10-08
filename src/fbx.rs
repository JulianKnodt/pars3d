use super::mesh::Mesh;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use fbxcel_dom::any::AnyDocument;

/// Load an FBX file from a given path.
pub fn load<P>(path: P) -> io::Result<Scene>
where
    P: AsRef<std::path::Path>,
{
    let f = File::open(path)?;
    let f = BufReader::new(f);
    match AnyDocument::from_seekable_reader(reader).expect("Failed to load document") {
        AnyDocument::V7400(fbx_ver, doc) => {
            // You got a document. You can do what you want.
        }
        // `AnyDocument` is nonexhaustive.
        // You should handle unknown document versions case.
        _ => panic!("Got FBX document of unsupported version"),
    }
}
