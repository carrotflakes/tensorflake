use std::ops::Mul;

use crate::*;

#[derive(Clone)]
pub struct MomentumSGDOptimizer {
    momentum: f32,
}

pub struct State {
    velocity: NDArray,
}

impl MomentumSGDOptimizer {
    pub fn new(momentum: f32) -> Self {
        MomentumSGDOptimizer { momentum }
    }
}

impl Optimizer for MomentumSGDOptimizer {
    type State = State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        State {
            velocity: NDArray::zeros(shape),
        }
    }

    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray, lr: f32) {
        tensor.cut_chain();
        state.velocity *= self.momentum;
        state.velocity += &grad.mul(scalar(-lr));
        *tensor = &*tensor + &state.velocity.clone().into();
    }
}

#[test]
fn test() {
    super::test_optimizer(MomentumSGDOptimizer::new(0.9), 0.01);
}
