mod gru;
mod lstm;

pub use gru::Gru;
pub use lstm::Lstm;

use crate::ComputedNDA;

pub trait Cell {
    type State: Clone + 'static;

    fn initial_state(&self, batch_size: usize) -> Self::State;
    fn get_input_size(&self) -> usize;
    fn step(&self, x: ComputedNDA, state: Self::State) -> (Self::State, ComputedNDA);

    fn encode(
        &self,
        initial_state: Self::State,
        x: &Vec<ComputedNDA>,
    ) -> (Self::State, Vec<ComputedNDA>) {
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
        mut state: Self::State,
        mut input: ComputedNDA,
        output_fn: impl Fn(ComputedNDA) -> ComputedNDA,
        output_to_input_fn: impl Fn(ComputedNDA) -> ComputedNDA,
        len: usize,
    ) -> Vec<ComputedNDA> {
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
