use crate::{
    functions::{Mul, Neg, Pow},
    Function, Tensor, Variable,
};

pub struct Div;

impl Function for Div {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 2);
        assert_eq!(xs[0].shape(), xs[1].shape());

        vec![&*xs[0] / &*xs[1]]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        vec![
            Div.call(vec![gys[0].clone(), gys[1].clone()])
                .pop()
                .unwrap(),
            Mul.call(vec![
                gys[0].clone(),
                Div.call(vec![
                    Neg.call(vec![xs[0].clone()]).pop().unwrap(),
                    Pow::new(2.0).call(vec![xs[1].clone()]).pop().unwrap(),
                ])
                .pop()
                .unwrap(),
            ])
            .pop()
            .unwrap(),
        ]
        // let mut gx0 = gys[0].data.clone();
        // let mut gx1 = gys[1].data.clone();
        // let x0 = &xs[0].data;
        // let x1 = &xs[1].data;
        // for i in 0..gx0.len() {
        //     gx0[i] = gx0[i] / x1[i];
        //     gx1[i] = gx1[i] * (-x0[i] / x1[i].powi(2));
        // }
        // vec![
        //     Variable::new(Tensor::new(gx0, &gys[0].shape())),
        //     Variable::new(Tensor::new(gx1, &gys[0].shape())),
        // ]
    }
}
