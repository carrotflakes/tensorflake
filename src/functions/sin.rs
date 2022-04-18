use super::{Mul, Neg};
use crate::*;

pub fn sin(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.sin()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sin",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].cos();
            vec![gx]
        },
    );

    y
}

pub fn cos(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.cos()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "cos",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &-&xs[0].sin();
            vec![gx]
        },
    );

    y
}

pub struct Sin;

impl Function for Sin {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.sin()).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        Mul.call(vec![gys[0].clone(), Cos.call(xs.clone()).pop().unwrap()])
    }
}

pub struct Cos;

impl Function for Cos {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.cos()).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        Mul.call(vec![
            gys[0].clone(),
            Neg.call(Sin.call(xs.clone())).pop().unwrap(),
        ])
    }
}
