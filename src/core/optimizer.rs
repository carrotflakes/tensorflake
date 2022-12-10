pub trait Optimizer<T: Sync + Send + 'static>: Sync + Send + 'static {
    type State: Sync + Send + 'static;

    fn new_state(&self, shape: &[usize]) -> Self::State;
    fn update(&mut self, data: &mut T, state: &mut Self::State, grad: &T);
}
