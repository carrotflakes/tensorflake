use super::NDArray;

pub trait Optimizer: Sync + Send + 'static {
    type State: Sync + Send + 'static;

    fn new_state(&self, shape: &[usize]) -> Self::State;
    fn update(&mut self, data: &mut NDArray, state: &mut Self::State, grad: &NDArray);
}
