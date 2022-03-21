use ndarray::Array;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::*;

pub struct Dropout {
    pub rate_fn: Box<dyn Fn() -> f32>,
}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        assert!(0.0 < rate && rate < 1.0);
        Self {
            rate_fn: Box::new(move || rate),
        }
    }

    pub fn from_rate_fn(rate_fn: Box<dyn Fn() -> f32>) -> Self {
        Self { rate_fn }
    }
}

impl Layer for Dropout {
    fn call(&self, xs: Vec<Variable>, train: bool) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        if !train {
            return xs;
        }
        assert_eq!(xs.len(), 1);
        let x = xs[0].clone();
        let rate = (self.rate_fn)();
        // TODO: use seed
        let fuctor = Variable::new(
            Array::random(x.shape(), Uniform::new(0.0, 1.0))
                .map(|x| if *x > rate { 1.0 / (1.0 - rate) } else { 0.0 })
                .into_tensor(),
        );
        vec![call!(functions::Mul, x, fuctor)]
    }

    fn all_params(&self) -> Vec<Variable> {
        vec![]
    }
}

#[test]
fn test() {
    let x = Variable::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_tensor());
    let ys = Dropout::new(0.8).call(vec![x.clone()], true);
    assert_eq!(&ys[0].shape(), &[2, 3]);
    dbg!(&*ys[0]);
}
