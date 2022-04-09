use std::ops::{DivAssign, Sub};

use ndarray::Axis;

use crate::functions::*;
use crate::*;

pub fn softmax(x: &Tensor) -> Tensor {
    let ndim = x.ndim();
    let x_max = x.map_axis(Axis(ndim - 1), |x| {
        *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    });
    let y = &**x - x_max.insert_axis(Axis(ndim - 1));
    let mut y = y.map(|x| x.exp());
    y.div_assign(&y.sum_axis(Axis(ndim - 1)).insert_axis(Axis(ndim - 1)));
    let y = Tensor::new(y.into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "softmax",
        move |xs, ys, gys| {
            let gx = &ys[0] * &gys[0];
            let sum_dx = gx.sum([xs[0].ndim() - 1], true);
            let gx = gx - &ys[0] * &sum_dx;
            vec![gx]
        },
    );

    y
}

pub struct Softmax;

impl Function for Softmax {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        let ndim = x.ndim();
        let x_max = x.map_axis(Axis(ndim - 1), |x| {
            *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        });
        let y = x.sub(x_max.insert_axis(Axis(ndim - 1)));
        let mut y = y.map(|x| x.exp());
        y.div_assign(&y.sum_axis(Axis(ndim - 1)).insert_axis(Axis(ndim - 1)));
        vec![y.into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        let gx = call!(Mul, ys[0], gys[0]);
        let sum_dx = call!(Sum::new(vec![xs[0].ndim() - 1], true), gx);
        vec![call!(Sub, gx, call!(Mul, ys[0], sum_dx))]
    }
}

#[test]
fn test() {
    let x = backprop(ndarray::array![[0.1, 0.2, 0.3], [0.0, 0.0, 100.0]].into_ndarray());
    let y = call!(Softmax, x.clone());
    dbg!(&*y);

    let grads = gradients(&[y], &[x], false);
    dbg!(&*grads[0]);
}
