use super::{Mul, Neg};
use crate::*;

pub struct Sin;

impl Function for Sin {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.sin()).into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Mul.call(vec![gys[0].clone(), Cos.call(xs.clone()).pop().unwrap()])
    }
}

pub struct Cos;

impl Function for Cos {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.cos()).into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Mul.call(vec![
            gys[0].clone(),
            Neg.call(Sin.call(xs.clone())).pop().unwrap(),
        ])
    }
}
