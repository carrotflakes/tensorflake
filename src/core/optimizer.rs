use super::{NDArray, Computed};

pub trait Optimizer: Sync + Send + 'static {
    type State: Sync + Send + 'static;

    fn new_state(&self, shape: &[usize]) -> Self::State;
    fn update(&mut self, tensor: &mut Computed, state: &mut Self::State, grad: &NDArray);
}
