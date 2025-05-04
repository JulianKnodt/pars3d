use super::F;

/// cheap noise function which is a bit stupid.
pub fn cheap_noise(i: F) -> F {
    ((i * 697.341 + 0.34).cos() + 1.) / 2.
}

// TODO add other kinds of noise here (blue noise, perlin, structured)
