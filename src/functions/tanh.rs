use crate::{
    functions::{Mul, Sub},
    *,
};

pub fn tanh(x: &Computed) -> Computed {
    let y = Computed::new(x.map(|x| x.tanh()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "tanh",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(Computed::new(scalar(1.0)) - ys[0].pow(2.0));
            vec![gx]
        },
    );

    y
}

pub struct Tanh;

impl Function for Tanh {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.tanh()).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            call!(Sub, Computed::new(scalar(1.0)), call!(Mul, ys[0], ys[0]))
        )]
    }
}
