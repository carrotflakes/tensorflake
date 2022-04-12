use crate::functions::*;
use crate::*;

pub trait Regularizer: Clone + Sync + Send + 'static {
    fn call(&self, x: &Tensor) -> Tensor;
}

#[derive(Clone)]
pub struct L1 {
    pub l1: f32,
}

impl L1 {
    pub fn new(l1: f32) -> Self {
        Self { l1 }
    }
}

impl Regularizer for L1 {
    fn call(&self, input: &Tensor) -> Tensor {
        abs(&input).sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l1).into()
    }
}

#[derive(Clone)]
pub struct L2 {
    pub l2: f32,
}

impl L2 {
    pub fn new(l2: f32) -> Self {
        Self { l2 }
    }
}

impl Regularizer for L2 {
    fn call(&self, input: &Tensor) -> Tensor {
        input.pow(2.0).sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l2).into()
    }
}

#[derive(Clone)]
pub struct L1L2 {
    pub l1: f32,
    pub l2: f32,
}

impl L1L2 {
    pub fn new(l1: f32, l2: f32) -> Self {
        Self { l1, l2 }
    }
}

impl Regularizer for L1L2 {
    fn call(&self, input: &Tensor) -> Tensor {
        abs(&input).sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l1).into()
            + input.pow(2.0).sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l2).into()
    }
}

#[test]
fn test() {
    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        optimizers::SGDOptimizer::new(1.0),
    );
    let l1 = L1::new(1.0);
    let loss = l1.call(&p.get_tensor());
    optimize(&loss);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0., 1., 2.].into_ndarray()
    );

    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        optimizers::SGDOptimizer::new(1.0),
    );
    let l2 = L2::new(0.25);
    let loss = l2.call(&p.get_tensor());
    optimize(&loss);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0.5, 1.0, 1.5].into_ndarray()
    );
}
