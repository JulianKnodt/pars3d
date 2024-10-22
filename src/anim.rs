use super::F;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationKind {
    #[default]
    Linear,
    CubicSpline,
    Step,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Translation,
    Rotation,
    Scale,
    MorphTargetWeights,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum OutputProperty {
    #[default]
    None,
    Translation(Vec<[F; 3]>),
    Rotation(Vec<[F; 4]>),
    Scale(Vec<[F; 3]>),
    MorphTargetWeight(Vec<F>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Channel {
    pub target_node_idx: usize,
    pub target_property: Property,
    pub sampler: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sampler {
    pub interpolation_kind: InterpolationKind,
    // Time to modify property
    pub input: Vec<F>,
    // What property is being modified
    pub output: OutputProperty,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    pub samplers: Vec<Sampler>,
}
