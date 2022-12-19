use crate::{IntoNDArray, NDArray};

pub fn positional_encoding(size: usize, depth: usize) -> NDArray {
    ndarray::Array2::from_shape_fn((size, depth), |(pos, d)| {
        if d % 2 == 0 {
            (pos as f32 / 10000.0f32.powf(d as f32 / depth as f32)).sin()
        } else {
            (pos as f32 / 10000.0f32.powf((d - 1) as f32 / depth as f32)).cos()
        }
    })
    .into_ndarray()
}

#[test]
fn test() {
    dbg!(positional_encoding(4, 3));
}
