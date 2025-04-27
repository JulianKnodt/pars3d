use crate::{Mesh, Scene};

impl From<super::VRML> for Scene {
  fn from(s: super::VRML) -> Scene {
    let mut scene = Scene::default();
    for g in s.groups {
      for c in g.children {
        scene.meshes.push(c.shape.into());
      }
    }
    scene
  }
}

impl From<super::Shape> for Mesh {
    fn from(s: super::Shape) -> Mesh {
        let super::Shape {
            points: v,
            indices: f,
            ..
        } = s;
        Mesh {
            v,
            f,
            ..Default::default()
        }
    }
}
