use crate::{call, functions::Mul, Function, Tensor, Variable};

pub struct Relu;

impl Function for Relu {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.max(0.0))]
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
            Variable::new(xs[0].map(|x| if *x > 0.0 { 1.0 } else { 0.0 }))
        )]
    }
}
