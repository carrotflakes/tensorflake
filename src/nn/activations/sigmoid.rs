use crate::{
    functions::{Mul, Sub},
    *,
};

pub fn sigmoid(x: &Computed) -> Computed {
    let y = Computed::new(x.map(|x| (x * 0.5).tanh() * 0.5 + 0.5).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sigmoid",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(&Computed::new(scalar(1.0)) - &ys[0]) * ys[0].clone();
            vec![gx]
        },
    );

    y
}

pub struct Sigmoid;

impl Function for Sigmoid {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);

        vec![xs[0]
            .map(|x| (x * 0.5).tanh() * 0.5 + 0.5)
            .into_ndarray()
            .into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            ys[0],
            call!(Sub, Computed::new(scalar(1.0)), ys[0])
        )]
    }
}

pub fn naive_sigmoid(x: Computed) -> Computed {
    Computed::new(scalar(1.0)) / (Computed::new(scalar(1.0)) + (-x).exp())
}
