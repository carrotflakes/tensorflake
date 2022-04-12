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
        kernel: &mut impl initializers::Initializer,
    ) -> Self {
        Self {
            input_size,
            state_size,
            ws: [
                kernel.initialize(&[input_size, state_size]),
                kernel.initialize(&[input_size, state_size]),
                kernel.initialize(&[input_size, state_size]),
            ],
            us: [
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
            ],
            bs: [
                kernel.initialize(&[state_size]),
                kernel.initialize(&[state_size]),
                kernel.initialize(&[state_size]),
            ],
        }
    }

    pub fn all_params(&self) -> Vec<Param> {
        self.ws
            .iter()
            .chain(self.us.iter())
            .chain(self.bs.iter())
            .cloned()
            .collect()
    }
}

impl Cell for Gru {
    fn get_state_size(&self) -> usize {
        self.state_size
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn step(&self, x: Tensor, state: Tensor) -> (Tensor, Tensor) {
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
        let state = (Tensor::new(NDArray::ones(z.shape())) - z.clone()) * state.clone()
            + z * tanh(
                &(x.matmul(&self.ws[2].get_tensor())
                    + (r * state).matmul(&self.us[2].get_tensor())
                    + self.bs[2].get_tensor()),
            );
        (state.clone(), state)
    }
}
