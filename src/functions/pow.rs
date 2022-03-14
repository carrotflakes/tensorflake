use super::Mul;
use crate::{Function, Tensor, Variable};

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

impl Function for Pow {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.powf(self.0))]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        Mul.call(vec![
            Pow::new(self.0 - 1.0)
                .call(vec![xs[0].clone()])
                .pop()
                .unwrap(),
            gys[0].clone(),
            Variable::new(ndarray::arr0(self.0).into_dyn()),
        ])
    }
}

#[test]
fn test_pow() {
    let a = Variable::<true>::new(ndarray::arr0(5.0).into_dyn());
    let ys = Pow(2.0).call(vec![a.clone()]);
    assert_eq!(*ys[0], ndarray::arr0(25.0).into_dyn());

    ys[0].set_grad(Variable::<true>::new(ndarray::arr0(1.0).into_dyn()));
    ys[0].backward(false, false);
    assert_eq!(
        *a.get_grad::<false>().unwrap(),
        ndarray::arr0(10.0).into_dyn()
    );
}
