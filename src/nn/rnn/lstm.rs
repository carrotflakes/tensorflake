use crate::{
    nn::activations::{sigmoid, tanh},
    *,
};

use super::Cell;

pub struct Lstm {
    pub input_size: usize,
    pub state_size: usize,
    pub ws: [Param; 4],
    pub us: [Param; 4],
    pub bs: [Param; 4],
}

impl Lstm {
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
                kernel.initialize(&[input_size, state_size]),
            ],
            us: [
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
            ],
            bs: [
                kernel.initialize(&[state_size]),
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

impl Cell for Lstm {
    type State = [Tensor; 2];

    fn initial_state(&self, batch_size: usize) -> Self::State {
        [
            Tensor::new(NDArray::zeros(&[batch_size, self.state_size][..])),
            Tensor::new(NDArray::zeros(&[batch_size, self.state_size][..])),
        ]
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn step(&self, x: Tensor, state: Self::State) -> (Self::State, Tensor) {
        let [c, h] = state;
        let f = sigmoid(
            &(x.matmul(&self.ws[0].get_tensor())
                + h.matmul(&self.us[0].get_tensor())
                + self.bs[0].get_tensor()),
        );
        let i = sigmoid(
            &(x.matmul(&self.ws[1].get_tensor())
                + h.matmul(&self.us[1].get_tensor())
                + self.bs[1].get_tensor()),
        );
        let o = tanh(
            &(x.matmul(&self.ws[2].get_tensor())
                + h.matmul(&self.us[2].get_tensor())
                + self.bs[2].get_tensor()),
        );
        let d = sigmoid(
            &(x.matmul(&self.ws[3].get_tensor())
                + h.matmul(&self.us[3].get_tensor())
                + self.bs[3].get_tensor()),
        );
        let c = f * c + i * d;
        let h = o * tanh(&c);
        ([c, h.clone()], h)
    }
}
