use std::ops::Mul;

use crate::*;

#[derive(Clone)]
pub struct SGDOptimizer {
    pub learning_rate: f32,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        SGDOptimizer { learning_rate }
    }
}

impl Optimizer for SGDOptimizer {
    type State = ();

    fn new_state(&self, shape: &[usize]) -> Self::State {
        drop(shape);
        ()
    }

    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray) {
        drop(state);
        tensor.cut_chain();
        *tensor = &*tensor + &grad.mul(scalar(-self.learning_rate)).into();
    }
}

#[test]
fn test() {
    super::test_optimizer(SGDOptimizer::new(0.01));
}
