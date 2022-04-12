mod gru;

pub use gru::Gru;

use crate::Tensor;

pub trait Cell {
    fn get_state_size(&self) -> usize;
    fn get_input_size(&self) -> usize;
    fn step(&self, x: Tensor, state: Tensor) -> (Tensor, Tensor);

    fn encode(&self, initial_state: Tensor, x: &Vec<Tensor>) -> (Tensor, Vec<Tensor>) {
        let mut state = initial_state.clone();
        let mut outputs = vec![];
        for x in x {
            let output;
            (state, output) = self.step(x.clone(), state);
            outputs.push(output.clone());
        }
        (state, outputs)
    }

    fn decode(
        &self,
        mut state: Tensor,
        mut input: Tensor,
        output_fn: impl Fn(Tensor) -> Tensor,
        output_to_input_fn: impl Fn(Tensor) -> Tensor,
        len: usize,
    ) -> Vec<Tensor> {
        let mut outputs = vec![];
        for _ in 0..len {
            let output;
            (state, output) = self.step(input, state);
            let output = output_fn(output);
            outputs.push(output.clone());
            input = output_to_input_fn(output);
        }
        outputs
    }
}
