use crate::{
    call,
    functions::{sum_to_axes_to_desire, Mul, Neg, Pow, SumTo},
    Function, Tensor, Variable,
};

pub struct Div;

impl Function for Div {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 2);

        vec![&*xs[0] / &*xs[1]]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        let mut gx0 = Div.call(vec![gys[0].clone(), xs[0].clone()]).pop().unwrap();

        let mut gx1 = Mul
            .call(vec![
                gys[0].clone(),
                Div.call(vec![
                    Neg.call(vec![xs[0].clone()]).pop().unwrap(),
                    Pow::new(2.0).call(vec![xs[1].clone()]).pop().unwrap(),
                ])
                .pop()
                .unwrap(),
            ])
            .pop()
            .unwrap();

        // fit shape
        if xs[0].shape() != gx0.shape() {
            gx0 = call!(
                SumTo::new(sum_to_axes_to_desire(gx0.shape(), xs[0].shape())),
                gx0
            );
        }

        if xs[1].shape() != gx1.shape() {
            gx1 = call!(
                SumTo::new(sum_to_axes_to_desire(gx1.shape(), xs[0].shape())),
                gx1
            );
        }

        vec![gx0, gx1]
    }
}
