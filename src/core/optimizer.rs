use super::{NDArray, Tensor};

pub trait Optimizer: Clone + Sync + Send + 'static {
    type State: Sync + Send + 'static;

    fn new_state(&self, shape: &[usize]) -> Self::State;
    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray);
}
