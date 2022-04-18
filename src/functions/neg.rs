use crate::*;

pub fn neg(x: &Computed) -> Computed {
    let y = Computed::new((-&**x).into_ndarray());

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
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| -x).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        Neg.call(gys.clone())
    }
}
