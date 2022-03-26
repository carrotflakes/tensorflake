use ndarray::Array;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::*;

pub struct Dropout {
    pub rate_fn: Box<dyn Fn() -> f32 + Send + Sync>,
}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        assert!(0.0 < rate && rate < 1.0);
        Self {
            rate_fn: Box::new(move || rate),
        }
    }

    pub fn from_rate_fn(rate_fn: Box<dyn Fn() -> f32 + Send + Sync>) -> Self {
        Self { rate_fn }
    }
}

impl Layer for Dropout {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output {
        if !train {
            return x;
        }
        let rate = (self.rate_fn)();
        // TODO: use seed
        let fuctor = Tensor::new(
            Array::random(x.shape(), Uniform::new(0.0, 1.0))
                .map(|x| if *x > rate { 1.0 / (1.0 - rate) } else { 0.0 })
                .into_ndarray(),
        );
        call!(functions::Mul, x, fuctor)
    }

    fn all_params(&self) -> Vec<Param> {
        vec![]
    }
}

#[test]
fn test() {
    let x = Tensor::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
    let y = Dropout::new(0.8).call(x.clone(), true);
    assert_eq!(&y.shape(), &[2, 3]);
    dbg!(&*y);
}
