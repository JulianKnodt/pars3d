use pars3d::{F, parse_args};

fn main() -> std::io::Result<()> {
    let args = parse_args!(
      "[Brighten Vertex Colors]: Brighten Vertex Colors:",
      Input("-i", "--input"; "Input Mesh"; |v: &str| !v.is_empty()) => input : String = String::new(),
      Output("-o", "--output"; "Output Mesh"; |v: &str| !v.is_empty()) => output : String = String::new(),
      Gamma("-g", "--gamma"; "Gamma to change brightness by") => gamma : F = 1./1.4,
      Sigmoid("-s", "--sigmoid"; "Apply sigmoid first") => sigmoid : bool = false => true,
    );

    let mut input = pars3d::load(args.input)?;

    for m in input.meshes.iter_mut() {
        for vc in m.vert_colors.iter_mut() {
            if args.sigmoid {
                *vc = vc.map(sigmoid);
            }
            *vc = vc.map(|c| c.powf(args.gamma));
            if args.sigmoid {
                *vc = vc.map(inv_sigmoid);
            }
        }
    }

    pars3d::save(&args.output, &input)
}

fn sigmoid(x: F) -> F {
    1. / (1. + (-x).exp())
}

fn inv_sigmoid(y: F) -> F {
    y.ln() - (1. - y).ln()
}
