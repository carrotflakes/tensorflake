/// SimpleGate
///
/// Simple Baselines for Image Restoration https://arxiv.org/abs/2204.04676
use ndarray::{azip, Array, Axis};

use crate::*;

pub fn simple_gate(x: &Tensor, axis: usize) -> Tensor {
    let (a, b) = x.view().split_at(Axis(axis), x.shape()[axis] / 2);
    let mut y = NDArray::zeros(a.shape());
    azip!((a in &a, b in &b, c in &mut y) *c = a * b);
    let y = Tensor::new(y);

    chain(
        &[x.clone()],
        &[y.clone()],
        true,
        "simple_gate",
        move |xs, _ys, gys| {
            let x = &xs[0];
            let mut gx = Array::zeros(x.shape());
            let (a, b) = x.view().split_at(Axis(axis), x.shape()[axis] / 2);
            let (ga, gb) = gx.view_mut().split_at(Axis(axis), x.shape()[axis] / 2);

            azip!((a in &*gys[0], b in &b, g in ga) *g =a * b);
            azip!((a in &*gys[0], b in &a, g in gb) *g =a * b);

            vec![gx.into_ndarray().into()]
        },
    );

    y
}
