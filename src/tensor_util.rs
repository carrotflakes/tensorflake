use ndarray::prelude::*;

use crate::*;

pub fn as_2d(tensor: &Tensor) -> ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>> {
    let shape = tensor.shape();
    tensor
        .view()
        .into_shape([
            shape.iter().take(shape.len() - 1).product(),
            *shape.last().unwrap(),
        ])
        .unwrap()
}
