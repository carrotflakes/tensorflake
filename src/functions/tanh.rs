use crate::{
    functions::{Mul, Sub},
    *,
};

pub struct Tanh;

impl Function for Tanh {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.tanh()).into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            call!(Sub, Tensor::new(scalar(1.0)), call!(Mul, ys[0], ys[0]))
        )]
    }
}
