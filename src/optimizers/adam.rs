use crate::*;

const EPS: f32 = 1e-8;

#[derive(Clone)]
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
}

pub struct State {
    mom: NDArray, // TODO: owned mom and vel
    vel: NDArray,
}

impl Adam {
    pub fn new() -> Self {
        Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    pub fn new_with_params(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
        }
    }
}

impl Optimizer for Adam {
    type State = State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        State {
            mom: NDArray::zeros(shape),
            vel: NDArray::zeros(shape),
        }
    }

    fn update(&mut self, data: &mut NDArray, state: &mut Self::State, grad: &NDArray) {
        state.mom = (&state.mom * self.beta1 + grad * (1.0 - self.beta1)).into_ndarray();
        state.vel =
            (&state.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_ndarray();
        *data = (&*data + &state.mom / state.vel.map(|x| x.sqrt() + EPS) * -self.learning_rate)
            .into_ndarray()
            .into();
    }
}

#[test]
fn test() {
    super::test_optimizer(Adam::new_with_params(0.01, 0.9, 0.999));
}
