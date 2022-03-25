use super::Mul;
use crate::*;

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

impl Function for Pow {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.powf(self.0)).into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Mul.call(vec![
            Pow::new(self.0 - 1.0)
                .call(vec![xs[0].clone()])
                .pop()
                .unwrap(),
            gys[0].clone(),
            Tensor::new(scalar(self.0)),
        ])
    }
}

#[test]
fn test_pow() {
    let a = backprop(scalar(5.0));
    let ys = Pow(2.0).call(vec![a.clone()]);
    assert_eq!(*ys[0], scalar(25.0));

    let grads = gradients(&ys, &[a.clone()], false);
    assert_eq!(*grads[0], scalar(10.0));
}
