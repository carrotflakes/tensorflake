use super::Mul;
use crate::{Function, Tensor, Variable};

pub struct Exp;

impl Function for Exp {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| x.exp())]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        Mul.call(vec![gys[0].clone(), Exp.call(xs.clone()).pop().unwrap()])
        // vec![Variable::new(gys[0].multiply(&xs[0].map(|x| x.exp())))]
    }
}
