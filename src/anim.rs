use super::F;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationKind {
    #[default]
    Linear,
    CubicSpline,
    Step,
}

impl InterpolationKind {
    pub fn apply(self, t: F) -> [F; 2] {
        match self {
            InterpolationKind::Step => [1., 0.],
            InterpolationKind::Linear => [1. - t, t],
            InterpolationKind::CubicSpline => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    XYZ,
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Translation(Dim),
    Rotation(Dim),
    Scale(Dim),
    MorphTargetWeights,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum OutputProperty {
    #[default]
    None,
    /// GLB
    Translation(Vec<[F; 3]>),
    Rotation(Vec<[F; 4]>),
    Scale(Vec<[F; 3]>),
    MorphTargetWeight(Vec<F>),

    /// FBX
    SingleChannel(Property, Vec<F>),
}

impl OutputProperty {
    pub fn len(&self) -> usize {
        match self {
            OutputProperty::None => 0,
            OutputProperty::Translation(t) => t.len(),
            OutputProperty::Rotation(t) => t.len(),
            OutputProperty::Scale(t) => t.len(),
            OutputProperty::MorphTargetWeight(t) => t.len(),
            OutputProperty::SingleChannel(_, t) => t.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Channel {
    pub target_node_idx: usize,
    pub target_property: Property,
    pub sampler: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Time {
    Float(F),
    Frame(u64),
}

impl Time {
    pub fn to_float(&self) -> F {
        match *self {
            Time::Float(f) => f,
            // TODO need to add a conversion method here
            Time::Frame(f) => f as F,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sampler {
    pub interpolation_kind: InterpolationKind,
    // Time to modify property
    pub input: Vec<Time>,
    // What property is being modified
    pub output: OutputProperty,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    pub samplers: Vec<Sampler>,
}
