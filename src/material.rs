use crate::gltf::GLTFMaterial;

/// Source material from loaded mesh.
#[derive(Debug, Clone, PartialEq)]
pub enum Material {
    GLTF(GLTFMaterial),
}
