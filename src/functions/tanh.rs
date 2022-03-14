use crate::{
    call,
    functions::{Mul, Sub},
    scalar, Function, Tensor, Variable,
};

pub struct Tanh;

impl Function for Tanh {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.tanh())]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            call!(Sub, Variable::new(scalar(1.0)), call!(Mul, ys[0], ys[0]))
        )]
    }
}
