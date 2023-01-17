/// ZerO initialization
/// https://arxiv.org/pdf/2110.12661.pdf
use crate::{initializers::Initializer, ndarray_util::eye, IntoNDArray, NDArray};

#[derive(Clone, Copy)]
pub struct ZerOInitializer;

impl Initializer<NDArray> for ZerOInitializer {
    fn initialize(&self, shape: &[usize]) -> NDArray {
        assert_eq!(shape.len(), 2);
        let m = shape[0];
        let n = shape[1];

        if m <= n {
            eye([m, n])
        } else {
            let scale = 2.0f32.powi((m as f32).log2().ceil() as i32 / 2).recip();
            ndarray::Array2::from_shape_fn([m, n], |(i, j)| {
                scale * (-1.0f32).powi((i & j).count_ones() as i32)
            })
            .into_ndarray()
        }
    }
}

pub fn hadamard_denormed(size: usize) -> crate::NDArray {
    ndarray::Array2::from_shape_fn([size, size], |(i, j)| {
        (-1.0f32).powi((i & j).count_ones() as i32)
    })
    .into_ndarray()
}
