use crate::{
    nn::activations::{sigmoid, tanh},
    *,
};

use super::Cell;

pub struct Gru {
    pub input_size: usize,
    pub state_size: usize,
    pub ws: [ParamNDA; 3],
    pub us: [ParamNDA; 3],
    pub bs: [ParamNDA; 3],
}

impl Gru {
    pub fn new(
        input_size: usize,
        state_size: usize,
        kernel: impl initializers::Initializer<ParamNDA> + initializers::Scope,
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
    type Input = (ComputedNDA, ComputedNDA);
    type Output = ComputedNDA;

    fn call(&self, input: Self::Input, _train: bool) -> Self::Output {
        let (x, state) = input;
        let z = sigmoid(
            &(x.matmul(&self.ws[0].get()) + state.matmul(&self.us[0].get()) + self.bs[0].get()),
        );
        let r = sigmoid(
            &(x.matmul(&self.ws[1].get()) + state.matmul(&self.us[1].get()) + self.bs[1].get()),
        );
        let state = (ComputedNDA::new(NDArray::ones(z.shape())) - z.clone()) * state.clone()
            + z * tanh(
                &(x.matmul(&self.ws[2].get())
                    + (r * state).matmul(&self.us[2].get())
                    + self.bs[2].get()),
            );
        state
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        self.ws
            .iter()
            .chain(self.us.iter())
            .chain(self.bs.iter())
            .cloned()
            .collect()
    }
}

impl Cell for Gru {
    type State = ComputedNDA;

    fn initial_state(&self, batch_size: usize) -> Self::State {
        ComputedNDA::new(NDArray::zeros(&[batch_size, self.state_size][..]))
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn step(&self, x: ComputedNDA, state: Self::State) -> (Self::State, ComputedNDA) {
        let state = self.call((x, state), false);
        (state.clone(), state)
    }
}
