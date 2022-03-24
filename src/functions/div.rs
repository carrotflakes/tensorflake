use crate::{
    functions::{sum_axes_to_desire, Mul, Neg, Pow, Sum},
    *,
};

pub struct Div;

impl Function for Div {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 2);

        vec![(&*xs[0] / &*xs[1]).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
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
                Sum::new(sum_axes_to_desire(gx0.shape(), xs[0].shape()), false),
                gx0
            );
        }

        if xs[1].shape() != gx1.shape() {
            gx1 = call!(
                Sum::new(sum_axes_to_desire(gx1.shape(), xs[0].shape()), false),
                gx1
            );
        }

        vec![gx0, gx1]
    }
}
