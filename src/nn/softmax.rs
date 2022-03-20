use std::ops::{DivAssign, Sub};

use ndarray::Axis;

use crate::functions::*;
use crate::*;

pub struct Softmax;

impl Function for Softmax {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        let ndim = x.ndim();
        let x_max = x.map_axis(Axis(ndim - 1), |x| {
            *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        });
        let y = x.sub(x_max.insert_axis(Axis(ndim - 1)));
        let mut y = y.map(|x| x.exp());
        y.div_assign(&y.sum_axis(Axis(ndim - 1)).insert_axis(Axis(ndim - 1)));
        vec![y.into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        let gx = call!(Mul, ys[0], gys[0]);
        let sum_dx = call!(SumTo::new(vec![xs[0].ndim() - 1], true), gx);
        vec![call!(Sub, gx, call!(Mul, ys[0], sum_dx))]
    }
}

#[test]
fn test() {
    let x = backprop(ndarray::array![[0.1, 0.2, 0.3], [0.0, 0.0, 100.0]].into_tensor());
    let y = call!(Softmax, x.clone());
    dbg!(&*y);

    let grads = gradients(&[y], &[x], false);
    dbg!(&*grads[0]);
}
