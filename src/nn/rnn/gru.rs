use crate::{
    nn::activations::{sigmoid, tanh},
    *,
};

use super::Cell;

pub struct Gru {
    pub input_size: usize,
    pub state_size: usize,
    pub ws: [Param; 3],
    pub us: [Param; 3],
    pub bs: [Param; 3],
}

impl Gru {
    pub fn new(
        input_size: usize,
        state_size: usize,
        kernel: impl initializers::Initializer,
    ) -> Self {
        Self {
            input_size,
            state_size,
            ws: [
                kernel
                    .scope(format!("w_{}", 0))
                    .initialize(&[input_size, state_size]),
                kernel
                    .scope(format!("w_{}", 1))
                    .initialize(&[input_size, state_size]),
                kernel
                    .scope(format!("w_{}", 2))
                    .initialize(&[input_size, state_size]),
            ],
            us: [
                kernel
                    .scope(format!("u_{}", 0))
                    .initialize(&[state_size, state_size]),
                kernel
                    .scope(format!("u_{}", 1))
                    .initialize(&[state_size, state_size]),
                kernel
                    .scope(format!("u_{}", 2))
                    .initialize(&[state_size, state_size]),
            ],
            bs: [
                kernel.scope(format!("b_{}", 0)).initialize(&[state_size]),
                kernel.scope(format!("b_{}", 1)).initialize(&[state_size]),
                kernel.scope(format!("b_{}", 2)).initialize(&[state_size]),
            ],
        }
    }
}

impl Layer for Gru {
    type Input = (Computed, Computed);
    type Output = Computed;

    fn call(&self, input: Self::Input, _train: bool) -> Self::Output {
        let (x, state) = input;
        let z = sigmoid(
            &(x.matmul(&self.ws[0].get_tensor())
                + state.matmul(&self.us[0].get_tensor())
                + self.bs[0].get_tensor()),
        );
        let r = sigmoid(
            &(x.matmul(&self.ws[1].get_tensor())
                + state.matmul(&self.us[1].get_tensor())
                + self.bs[1].get_tensor()),
        );
        let state = (Computed::new(NDArray::ones(z.shape())) - z.clone()) * state.clone()
            + z * tanh(
                &(x.matmul(&self.ws[2].get_tensor())
                    + (r * state).matmul(&self.us[2].get_tensor())
                    + self.bs[2].get_tensor()),
            );
        state
    }

    fn all_params(&self) -> Vec<Param> {
        self.ws
            .iter()
            .chain(self.us.iter())
            .chain(self.bs.iter())
            .cloned()
            .collect()
    }
}

impl Cell for Gru {
    type State = Computed;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        Computed::new(NDArray::zeros(&[batch_size, self.state_size][..]))
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn step(&self, x: Computed, state: Self::State) -> (Self::State, Computed) {
        let state = self.call((x, state), false);
        (state.clone(), state)
    }
}
