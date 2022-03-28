use std::ops::Mul;

use crate::*;

#[derive(Clone)]
pub struct SGDOptimizer;

impl SGDOptimizer {
    pub fn new() -> Self {
        SGDOptimizer
    }
}

impl Optimizer for SGDOptimizer {
    type State = ();

    fn new_state(&self, shape: &[usize]) -> Self::State {
        drop(shape);
        ()
    }

    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray, lr: f32) {
        drop(state);
        tensor.cut_chain();
        *tensor = &*tensor + &grad.mul(scalar(-lr)).into();
    }
}

#[test]
fn test() {
    super::test_optimizer(SGDOptimizer::new(), 0.01);
}
