use super::{
    AnimCurveNodeKey, FBXAnimCurve, FBXAnimCurveNode, FBXScene,
    MeshOrNode, /*FBXAnimLayer, FBXAnimStack,*/
    NodeAnimAttrKey,
};
use crate::anim::{
    Animation, Channel, Dim, InterpolationKind, OutputProperty, Property, Sampler, Samplers, Time,
};

impl FBXScene {
    pub fn extract_animations(&self) -> impl Iterator<Item = Animation> + '_ {
        (0..self.anim_layers.len()).map(|i| self.extract_animation(i))
    }
    pub fn extract_animation(&self, layer_idx: usize) -> Animation {
        let a_l = &self.anim_layers[layer_idx];
        //let a_s = &self.anim_stacks[a_l.anim_stack];

        //let samplers =

        Animation {
            name: a_l.name.clone(),
            ..Default::default()
        }
    }

    pub fn channel_for_anim_curve_node(&self, a_cn_idx: usize) -> Channel {
        let a_cn = &self.anim_curve_nodes[a_cn_idx];
        let dim = match (a_cn.dx, a_cn.dy, a_cn.dz) {
            (Some(_), None, None) => Dim::X,
            (None, Some(_), None) => Dim::Y,
            (None, None, Some(_)) => Dim::Z,
            (Some(_), Some(_), Some(_)) => Dim::XYZ,
            _ => todo!(),
        };
        let target_property: Property = match a_cn.rel_key {
            NodeAnimAttrKey::Translation => Property::Translation(dim),
            NodeAnimAttrKey::Rotation => Property::Rotation(dim),
            NodeAnimAttrKey::Scaling => Property::Rotation(dim),
            x => todo!("{x:?}"),
        };
        let mut a_cs = self
            .anim_curves
            .iter()
            .enumerate()
            .filter(|v| v.1.anim_curve_node == a_cn_idx)
            .map(|(i, _)| i);
        let a_c0 = a_cs.next();
        let a_c1 = a_cs.next();
        let a_c2 = a_cs.next();
        assert_eq!(a_cs.next(), None);
        let sampler = match (a_c0, a_c1, a_c2) {
            (None, None, None) => todo!("Case where anim curve node has no curves"),
            (Some(i), None, None) => Samplers::One(i),
            (Some(i), Some(j), None) => Samplers::Two([i, j]),
            (Some(i), Some(j), Some(k)) => Samplers::Three([i, j, k]),
            _ => unreachable!(),
        };
        let MeshOrNode::Node(node_idx) = a_cn.rel else {
            todo!();
        };
        Channel {
            target_node_idx: node_idx,
            target_property,
            sampler,
        }
    }
}

impl From<(FBXAnimCurve, &[FBXAnimCurveNode])> for Sampler {
    fn from((a_c, a_cns): (FBXAnimCurve, &[FBXAnimCurveNode])) -> Sampler {
        let a_cn = &a_cns[a_c.anim_curve_node];
        let dim = || match a_c.anim_curve_node_key {
            AnimCurveNodeKey::X => Dim::X,
            AnimCurveNodeKey::Y => Dim::Y,
            AnimCurveNodeKey::Z => Dim::Z,
            AnimCurveNodeKey::DeformPercent => unreachable!(),
            AnimCurveNodeKey::None => todo!(),
        };
        let prop = match a_cn.rel_key {
            NodeAnimAttrKey::Translation => Property::Translation(dim()),
            NodeAnimAttrKey::Rotation => Property::Rotation(dim()),
            NodeAnimAttrKey::Scaling => Property::Scale(dim()),
            NodeAnimAttrKey::DeformPercent => unreachable!(),
            NodeAnimAttrKey::None => todo!(),
        };
        let output_property = OutputProperty::SingleChannel(prop, a_c.values);
        Sampler {
            interpolation_kind: InterpolationKind::Linear,
            input: a_c.times.into_iter().map(Time::Frame).collect(),
            output: output_property,
        }
    }
}
