use std::ops::Mul;

use crate::*;

#[derive(Clone)]
pub struct MomentumSGD {
    pub learning_rate: f32,
    pub momentum: f32,
}

pub struct State {
    velocity: NDArray,
}

impl MomentumSGD {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        MomentumSGD {
            learning_rate,
            momentum,
        }
    }
}

impl Optimizer<NDArray> for MomentumSGD {
    type State = State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        State {
            velocity: NDArray::zeros(shape),
        }
    }

    fn update(&mut self, data: &mut NDArray, state: &mut Self::State, grad: &NDArray) {
        state.velocity *= self.momentum;
        state.velocity += &grad.mul(scalar(-self.learning_rate));
        *data = (&*data + &state.velocity).into_ndarray();
    }
}

#[test]
fn test() {
    super::test_optimizer(MomentumSGD::new(0.01, 0.9));
}
