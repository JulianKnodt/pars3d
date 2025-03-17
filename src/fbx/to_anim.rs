use super::{
    AnimCurveNodeKey, FBXAnimCurve, FBXAnimCurveNode, FBXScene,
    NodeAnimAttrKey, /*FBXAnimLayer, FBXAnimStack,*/
};
use crate::anim::{Animation, Dim, InterpolationKind, OutputProperty, Property, Sampler, Time};

impl FBXScene {
    pub fn extract_animations(&self) -> impl Iterator<Item = Animation> + '_ {
        (0..self.anim_layers.len()).map(|i| self.extract_animation(i))
    }
    pub fn extract_animation(&self, layer_idx: usize) -> Animation {
        let a_l = &self.anim_layers[layer_idx];
        //let a_s = &self.anim_stacks[a_l.anim_stack];

        //let samplers =

        let anim = Animation {
            name: a_l.name.clone(),
            ..Default::default()
        };
        anim
    }
}

// There can be multiple different anim curves for each anim curve node.
/*
impl From<FBXAnimCurveNode> for Channel {
  fn from(a_cn: FBXAnimCurveNode) -> Channel {
    let target_property: Property = match a_cn.key {
      x => todo!("{x:?}"),
    }

    Channel {
      target_node_idx: a_cn.node,
      target_property,
      sampler: a_cn.anim_curve,
    }
  }
}
*/

impl From<(FBXAnimCurve, &[FBXAnimCurveNode])> for Sampler {
    fn from((a_c, a_cns): (FBXAnimCurve, &[FBXAnimCurveNode])) -> Sampler {
        let a_cn = &a_cns[a_c.anim_curve_node];
        let dim = || match a_c.anim_curve_node_key {
            AnimCurveNodeKey::X => Dim::X,
            AnimCurveNodeKey::Y => Dim::Y,
            AnimCurveNodeKey::Z => Dim::Z,
            AnimCurveNodeKey::UnknownDefault => todo!(),
            x => todo!("{x:?}"),
        };
        let prop = match a_cn.node_key {
            NodeAnimAttrKey::Translation => Property::Translation(dim()),
            NodeAnimAttrKey::Rotation => Property::Rotation(dim()),
            NodeAnimAttrKey::Scaling => Property::Scale(dim()),
            NodeAnimAttrKey::UnknownDefault => todo!(),
        };
        let output_property = OutputProperty::SingleChannel(prop, a_c.values);
        Sampler {
            interpolation_kind: InterpolationKind::Linear,
            input: a_c.times.into_iter().map(Time::Frame).collect(),
            output: output_property,
        }
    }
}
