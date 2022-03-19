use super::Mul;
use crate::*;

pub struct Exp;

impl Function for Exp {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);
        vec![(xs[0].map(|x| x.exp())).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        Mul.call(vec![gys[0].clone(), Exp.call(xs.clone()).pop().unwrap()])
    }
}
