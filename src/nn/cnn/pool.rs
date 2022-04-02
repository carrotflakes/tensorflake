use super::im2col::{get_conv_outsize, Im2col};
use crate::{functions::*, *};

pub fn naive_max_pooling(
    x: &Tensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    pad: [usize; 2],
) -> Tensor {
    let x_shape = x.shape();
    let [kh, kw] = kernel_size;
    let oh = get_conv_outsize(x_shape[2], kernel_size[0], stride[0], pad[0]);
    let ow = get_conv_outsize(x_shape[3], kernel_size[1], stride[1], pad[1]);

    let col = call!(Im2col::new(kernel_size, stride, pad, true), x);
    let col = col.reshape([col.shape().iter().product::<usize>() / (kh * kw), kh * kw]);
    let y = max(1, &col);
    call!(
        Transpose::new(vec![0, 3, 1, 2]),
        call!(Reshape::new(vec![x_shape[0], oh, ow, x_shape[1]]), y)
    )
}

#[test]
fn test_naive_max_pooling() {
    use ndarray::prelude::*;
    let x = backprop(
        Array::from_shape_vec((1, 3, 4, 4), (0..16 * 3).map(|x| (x % 7) as f32).collect())
            .unwrap()
            .into_ndarray(),
    );

    let y = naive_max_pooling(&x, [2, 2], [2, 2], [0, 0]);
    dbg!(&*y);

    let grads = gradients(&[y.clone()], &[x.clone()], true);
    dbg!(&*grads[0]);
}

pub fn naive_sum_pooling(
    x: &Tensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    pad: [usize; 2],
) -> Tensor {
    let x_shape = x.shape();
    let [kh, kw] = kernel_size;
    let oh = get_conv_outsize(x_shape[2], kernel_size[0], stride[0], pad[0]);
    let ow = get_conv_outsize(x_shape[3], kernel_size[1], stride[1], pad[1]);

    let col = call!(Im2col::new(kernel_size, stride, pad, true), x);
    let col = col.reshape([col.shape().iter().product::<usize>() / (kh * kw), kh * kw]);
    let y = call!(Sum::new(vec![1], false), col);
    call!(
        Transpose::new(vec![0, 3, 1, 2]),
        call!(Reshape::new(vec![x_shape[0], oh, ow, x_shape[1]]), y)
    )
}

#[test]
fn test_naive_sum_pooling() {
    use ndarray::prelude::*;
    let x = backprop(
        Array::from_shape_vec((1, 3, 4, 4), (0..16 * 3).map(|x| x as f32).collect())
            .unwrap()
            .into_ndarray(),
    );

    let y = naive_sum_pooling(&x, [2, 2], [2, 2], [0, 0]);
    dbg!(&*y);
}
