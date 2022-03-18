use crate::{
    functions::{Mul, Sub},
    *,
};

pub struct Tanh;

impl Function for Tanh {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.tanh()).into_tensor()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            call!(Sub, Variable::new(scalar(1.0)), call!(Mul, ys[0], ys[0]))
        )]
    }
}
