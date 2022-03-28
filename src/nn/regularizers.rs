use crate::functions::*;
use crate::*;

pub struct L1 {
    pub l1: f32,
}

impl L1 {
    pub fn new(l1: f32) -> Self {
        Self { l1 }
    }
}

impl Layer for L1 {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, input: Self::Input, _: bool) -> Self::Output {
        call!(Sum::new((0..input.ndim()).collect(), false), abs(&input)) * scalar(self.l1).into()
    }

    fn all_params(&self) -> Vec<Param> {
        vec![]
    }
}

pub struct L2 {
    pub l2: f32,
}

impl L2 {
    pub fn new(l2: f32) -> Self {
        Self { l2 }
    }
}

impl Layer for L2 {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, input: Self::Input, _: bool) -> Self::Output {
        call!(
            Sum::new((0..input.ndim()).collect(), false),
            call!(Pow::new(2.0), input)
        ) * scalar(self.l2).into()
    }

    fn all_params(&self) -> Vec<Param> {
        vec![]
    }
}

pub struct L1L2 {
    pub l1: f32,
    pub l2: f32,
}

impl L1L2 {
    pub fn new(l1: f32, l2: f32) -> Self {
        Self { l1, l2 }
    }
}

impl Layer for L1L2 {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, input: Self::Input, _: bool) -> Self::Output {
        call!(Sum::new((0..input.ndim()).collect(), false), abs(&input)) * scalar(self.l1).into()
            + call!(
                Sum::new((0..input.ndim()).collect(), false),
                call!(Pow::new(2.0), input)
            ) * scalar(self.l2).into()
    }

    fn all_params(&self) -> Vec<Param> {
        vec![]
    }
}

#[test]
fn test() {
    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        optimizers::SGDOptimizer,
    );
    let l1 = L1::new(1.0);
    let loss = l1.call(p.get_tensor(), true);
    optimize(&loss, 1.0);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0., 1., 2.].into_ndarray()
    );

    let p = Param::new(
        ndarray::array![1., 2., 3.].into_ndarray(),
        optimizers::SGDOptimizer,
    );
    let l2 = L2::new(0.25);
    let loss = l2.call(p.get_tensor(), true);
    optimize(&loss, 1.0);
    assert_eq!(
        &*p.get_tensor(),
        &ndarray::array![0.5, 1.0, 1.5].into_ndarray()
    );
}
