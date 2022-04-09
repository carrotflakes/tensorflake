use crate::{
    functions::{Mul, Sub},
    *,
};

pub fn sigmoid(x: &Tensor) -> Tensor {
    let y = Tensor::new(x.map(|x| (x * 0.5).tanh() * 0.5 + 0.5).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sigmoid",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(&Tensor::new(scalar(1.0)) - &ys[0]) * ys[0].clone();
            vec![gx]
        },
    );

    y
}

pub struct Sigmoid;

impl Function for Sigmoid {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0]
            .map(|x| (x * 0.5).tanh() * 0.5 + 0.5)
            .into_ndarray()
            .into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        vec![call!(
            Mul,
            gys[0],
            ys[0],
            call!(Sub, Tensor::new(scalar(1.0)), ys[0])
        )]
    }
}

pub fn naive_sigmoid(x: Tensor) -> Tensor {
    Tensor::new(scalar(1.0)) / (Tensor::new(scalar(1.0)) + (-x).exp())
}
