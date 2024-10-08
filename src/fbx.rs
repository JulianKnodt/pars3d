use super::mesh::{Node, Scene};

use std::fs::File;
use std::io::{self, BufReader};

use fbxcel_dom::any::AnyDocument;

/// Load an FBX file from a given path.
pub fn load<P>(path: P) -> io::Result<Scene>
where
    P: AsRef<std::path::Path>,
{
    let f = File::open(path)?;
    let f = BufReader::new(f);

    let doc = match AnyDocument::from_seekable_reader(f).expect("Failed to load document") {
        AnyDocument::V7400(_fbx_ver, doc) => doc,
        // `AnyDocument` is nonexhaustive.
        // You should handle unknown document versions case.
        _ => panic!("Got FBX document of unsupported version"),
    };

    let mut out = Scene::default();
    out.nodes.push(Node::default());

    macro_rules! get_object_by_parent {
        ($parent_id: expr) => {{
            doc.objects().filter(|o| {
                o.destination_objects()
                    .any(|v| v.object_id().raw() as usize == $parent_id)
            })
        }};
    }

    let convert_node = |id: usize, parent: usize, root_node: usize, out: &mut Scene| {
        let nexts = get_object_by_parent!(id).filter(|o| o.class() == "Model");

        for n in nexts {
            println!("{:?}", n.get_typed());
        }
    };

    convert_node(0, 0, 0, &mut out);

    Ok(out)
}

#[test]
fn test_load_fbx() {
    let _v = load("CGame_Test_w_Ske.fbx").unwrap();
}
