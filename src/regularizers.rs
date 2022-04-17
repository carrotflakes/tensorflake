use crate::*;

pub trait Regularizer: Sync + Send + 'static {
    fn loss(&self, x: &Tensor) -> Tensor;

    fn grad(&self, x: &Tensor) -> Tensor {
        let loss = self.loss(x);
        gradients(&[loss], &[x.clone()], false).pop().unwrap()
    }
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
    fn loss(&self, input: &Tensor) -> Tensor {
        input.abs().sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l1).into()
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
    fn loss(&self, input: &Tensor) -> Tensor {
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
    fn loss(&self, input: &Tensor) -> Tensor {
        input.abs().sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l1).into()
            + input.pow(2.0).sum(Vec::from_iter(0..input.ndim()), false) * scalar(self.l2).into()
    }
}

#[test]
fn test() {
    use crate::*;

    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        "param".into(),
        optimizers::SGDOptimizer::new(1.0),
    );
    let l1 = L1::new(1.0);
    let loss = l1.loss(&p.get_tensor());
    optimize(&loss);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0., 1., 2.].into_ndarray()
    );

    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        "param".into(),
        optimizers::SGDOptimizer::new(1.0),
    );
    let l2 = L2::new(0.25);
    let loss = l2.loss(&p.get_tensor());
    optimize(&loss);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0.5, 1.0, 1.5].into_ndarray()
    );

    let x = backprop(ndarray::array![-1., 1., 2.].into_ndarray());
    for r in &[
        &L1::new(1.1) as &dyn Regularizer,
        &L2::new(1.2),
        &L1L2::new(1.1, 1.2),
    ] {
        assert_eq!(
            &*r.grad(&x),
            &*gradients(&[r.loss(&x)], &[x.clone()], false)[0]
        );
    }
}
