use std::ops::Sub;

use ndarray::{Array1, Axis};

use crate::ndarray_util::onehot;
use crate::nn::activations::{relu, softmax};
use crate::*;

pub fn naive_mean_squared_error(x0: ComputedNDA, x1: ComputedNDA) -> ComputedNDA {
    let x = (x0 - x1).pow(2.0);
    x.sum(Vec::from_iter(0..x.ndim()), false)
        / ComputedNDA::new(scalar(x.shape().iter().product::<usize>() as f32))
}

pub fn softmax_cross_entropy(t: Vec<usize>, x: &ComputedNDA) -> ComputedNDA {
    let n = x.shape().iter().take(x.ndim() - 1).product();
    let log_z = log_sum_exp(&*x);
    let log_p = x.to_shape((n, x.shape()[x.ndim() - 1])).unwrap();
    let mut y = 0.0;
    for i in 0..n {
        y -= log_p[[i, t[i]]] - log_z[i];
    }
    let y = ComputedNDA::new(scalar(y / n as f32));

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "softmax_cross_entropy",
        move |xs, _ys, gys| {
            let n: usize = xs[0].shape().iter().take(xs[0].ndim() - 1).product();
            let class_num = xs[0].shape()[xs[0].ndim() - 1];
            let gy = &gys[0] * &ComputedNDA::new(scalar(1.0 / n as f32));
            let y = softmax(&xs[0]);
            let t_onehot = ComputedNDA::new(
                onehot(&Array1::from(t.clone()), class_num)
                    .into_shape(y.shape())
                    .unwrap(),
            );
            vec![(y - t_onehot) * gy]
        },
    );

    y
}

#[test]
fn test_softmax_cross_entropy() {
    let x = backprop(ndarray::array![[0.1, 0.2, 0.3], [0.0, 0.0, 100.0]].into_ndarray());
    let t = vec![1, 2];
    let loss = softmax_cross_entropy(t, &x);
    dbg!(&*loss);

    let grads = gradients(&[loss], &vec![x.clone()], false);
    dbg!(&*grads[0]);
}

pub fn softmax_cross_entropy_with_logits(
    labels: &ComputedNDA,
    logits: &ComputedNDA,
    axis: usize,
) -> ComputedNDA {
    let x = softmax(logits);
    let y = -(labels * &x.log()).sum([axis], false);

    chain(
        &[labels.clone(), logits.clone()],
        &[y.clone()],
        false,
        "softmax_cross_entropy_with_logits",
        move |xs, _ys, gys| {
            let labels = &xs[0];
            let gy = &gys[0];
            let mut shape = labels.shape().to_vec();
            shape[axis] = 1;
            let n: usize = shape.iter().product();

            let gy = gy.reshape(shape) / ComputedNDA::new(scalar(n as f32));
            let g_logits = &(&x - labels) * &gy;
            let g_labels = -(&x.log() * &gy);
            vec![g_labels, g_logits]
        },
    );

    y
}

#[test]
fn test_softmax_cross_entropy_with_logits() {
    let x = backprop(ndarray::array![[0.1, 0.2, 0.3], [0.0, 0.0, 100.0]].into_ndarray());
    let t = backprop(ndarray::array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_ndarray());
    let loss = softmax_cross_entropy_with_logits(&t, &x, 1);
    dbg!(&*loss);

    let grads = gradients(&[loss], &vec![x.clone()], false);
    dbg!(&*grads[0]);
}

pub fn sigmoid_cross_entropy_with_logits(labels: &ComputedNDA, logits: &ComputedNDA) -> ComputedNDA {
    relu(logits) - logits * labels + (ComputedNDA::new(scalar(1.0)) + (-logits.abs()).exp()).log()
}

// max(x) + log(sum(exp(x - max(x))))
pub fn log_sum_exp(x: &NDArray) -> NDArray {
    let ndim = x.ndim();
    let x_max = x.map_axis(Axis(ndim - 1), |x| {
        *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    });

    (&x_max
        + (x.sub(&x_max.view().insert_axis(Axis(ndim - 1))))
            .map(|x| x.exp())
            .sum_axis(Axis(ndim - 1))
            .map(|x| x.ln()))
    .into_ndarray()
}

#[test]
fn test_log_sum_exp() {
    let x = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],].into_ndarray();
    let y = log_sum_exp(&x);
    assert_eq!(y.shape(), &[2]);
    assert!(
        (y[0] - (3.0 + ((1.0f32 - 3.0).exp() + (2.0f32 - 3.0).exp() + (3.0f32 - 3.0).exp()).ln()))
            .abs()
            < 1e-6
    );
}
