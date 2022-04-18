mod gru;
mod lstm;

pub use gru::Gru;
pub use lstm::Lstm;

use crate::Computed;

pub trait Cell {
    type State: Clone + Sync + Send + 'static;

    fn initial_state(&self, batch_size: usize) -> Self::State;
    fn get_input_size(&self) -> usize;
    fn step(&self, x: Computed, state: Self::State) -> (Self::State, Computed);

    fn encode(&self, initial_state: Self::State, x: &Vec<Computed>) -> (Self::State, Vec<Computed>) {
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
        mut input: Computed,
        output_fn: impl Fn(Computed) -> Computed,
        output_to_input_fn: impl Fn(Computed) -> Computed,
        len: usize,
    ) -> Vec<Computed> {
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
