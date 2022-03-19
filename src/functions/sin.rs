use super::{Mul, Neg};
use crate::*;

pub struct Sin;

impl Function for Sin {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.sin()).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        Mul.call(vec![gys[0].clone(), Cos.call(xs.clone()).pop().unwrap()])
    }
}

pub struct Cos;

impl Function for Cos {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.cos()).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        Mul.call(vec![
            gys[0].clone(),
            Neg.call(Sin.call(xs.clone())).pop().unwrap(),
        ])
    }
}
