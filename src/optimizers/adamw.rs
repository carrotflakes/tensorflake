use crate::*;

const EPS: f32 = 1e-8;

#[derive(Clone)]
pub struct AdamWOptimizer {
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

pub struct State {
    mom: NDArray, // TODO: owned mom and vel
    vel: NDArray,
}

impl AdamWOptimizer {
    pub fn new() -> Self {
        AdamWOptimizer {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.00001,
        }
    }

    pub fn new_with_params(beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        AdamWOptimizer {
            beta1,
            beta2,
            weight_decay,
        }
    }
}

impl Optimizer for AdamWOptimizer {
    type State = State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        State {
            mom: NDArray::zeros(shape),
            vel: NDArray::zeros(shape),
        }
    }

    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray, lr: f32) {
        tensor.cut_chain();
        let wd = &**tensor * self.weight_decay;
        let grad = grad + &wd;
        state.mom = (&state.mom * self.beta1 + &grad * (1.0 - self.beta1)).into_ndarray();
        state.vel =
            (&state.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_ndarray();
        let mut a = &**tensor + &state.mom / state.vel.map(|x| x.sqrt() + EPS) * -lr + &wd;

        // Treat denormals as zero
        a = a.map(|x| if x.abs() < self.weight_decay { 0.0 } else { *x });

        *tensor = a.into_ndarray().into();
    }
}

#[test]
fn test() {
    super::test_optimizer(AdamWOptimizer::new(), 0.01);
}
