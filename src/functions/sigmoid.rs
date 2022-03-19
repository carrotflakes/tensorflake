use crate::{
    functions::{Mul, Sub},
    *,
};

pub struct Sigmoid;

impl Function for Sigmoid {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| (x * 0.5).tanh() * 0.5 + 0.5).into_tensor().into()]
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
            ys[0],
            call!(Sub, Variable::new(scalar(1.0)), ys[0])
        )]
    }
}
