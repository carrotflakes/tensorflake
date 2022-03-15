use crate::{call, Function, Tensor, Variable};

use super::{sum_to_axes_to_desire, SumTo};

pub struct Mul;

impl Function for Mul {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = y * &**x;
        }
        vec![y]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        xs.iter()
            .enumerate()
            .map(|(i, x)| {
                let mut g = Mul
                    .call(
                        (0..xs.len())
                            .filter(|j| *j != i)
                            .map(|j| xs[j].clone())
                            .chain(gys.iter().cloned())
                            .collect(),
                    )
                    .pop()
                    .unwrap();

                // fit shape
                if x.shape() != g.shape() {
                    g = call!(SumTo::new(sum_to_axes_to_desire(g.shape(), x.shape())), g);
                }

                g
            })
            .collect()
    }
}
