use crate::*;

pub fn neg(x: &Tensor) -> Tensor {
    let y = Tensor::new((-&**x).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "neg",
        move |_xs, _ys, gys| {
            let gx = -&gys[0];
            vec![gx]
        },
    );

    y
}

pub struct Neg;

impl Function for Neg {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| -x).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Neg.call(gys.clone())
    }
}
