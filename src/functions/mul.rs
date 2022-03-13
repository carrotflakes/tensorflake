use crate::{Function, Tensor, Variable};

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
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        (0..xs.len())
            .map(|i| {
                Mul.call(
                    (0..xs.len())
                        .filter(|j| *j != i)
                        .map(|j| xs[j].clone())
                        .chain(gys.iter().cloned())
                        .collect(),
                )
                .pop()
                .unwrap()
            })
            .collect()
    }
}
