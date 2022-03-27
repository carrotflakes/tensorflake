use std::sync::Mutex;

use ndarray::Array;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::Isaac64Rng;

use crate::*;

pub struct Dropout {
    pub rate_fn: Box<dyn Fn() -> f32 + Send + Sync>,
    rng: Mutex<Isaac64Rng>,
}

impl Dropout {
    pub fn new(rate: f32, seed: u64) -> Self {
        assert!(0.0 < rate && rate < 1.0);
        Self {
            rate_fn: Box::new(move || rate),
            rng: Mutex::new(Isaac64Rng::seed_from_u64(seed)),
        }
    }

    pub fn from_rate_fn(rate_fn: Box<dyn Fn() -> f32 + Send + Sync>, seed: u64) -> Self {
        Self {
            rate_fn,
            rng: Mutex::new(Isaac64Rng::seed_from_u64(seed)),
        }
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
            Array::random_using(
                x.shape(),
                Uniform::new(0.0, 1.0),
                &mut *self.rng.lock().unwrap(),
            )
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
    let y = Dropout::new(0.8, 42).call(x.clone(), true);
    assert_eq!(&y.shape(), &[2, 3]);
    dbg!(&*y);
}
