use super::Mul;
use crate::*;

pub fn exp(x: &Tensor) -> Tensor {
    let y = Tensor::new((**x).map(|x| x.exp()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "exp",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].exp();
            vec![gx]
        },
    );

    y
}

pub struct Exp;

impl Function for Exp {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![(xs[0].map(|x| x.exp())).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Mul.call(vec![gys[0].clone(), Exp.call(xs.clone()).pop().unwrap()])
    }
}
