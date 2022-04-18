use crate::*;

const EPS: f32 = 1e-8;

#[derive(Clone)]
pub struct AdamOptimizer {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
}

pub struct State {
    mom: NDArray, // TODO: owned mom and vel
    vel: NDArray,
}

impl AdamOptimizer {
    pub fn new() -> Self {
        AdamOptimizer {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    pub fn new_with_params(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        AdamOptimizer {
            learning_rate,
            beta1,
            beta2,
        }
    }
}

impl Optimizer for AdamOptimizer {
    type State = State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        State {
            mom: NDArray::zeros(shape),
            vel: NDArray::zeros(shape),
        }
    }

    fn update(&mut self, tensor: &mut Computed, state: &mut Self::State, grad: &NDArray) {
        tensor.unchain();
        state.mom = (&state.mom * self.beta1 + grad * (1.0 - self.beta1)).into_ndarray();
        state.vel =
            (&state.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_ndarray();
        *tensor = (&**tensor
            + &state.mom / state.vel.map(|x| x.sqrt() + EPS) * -self.learning_rate)
            .into_ndarray()
            .into();
    }
}

#[test]
fn test() {
    super::test_optimizer(AdamOptimizer::new_with_params(0.01, 0.9, 0.999));
}
