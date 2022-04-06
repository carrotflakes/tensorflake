use crate::*;

#[derive(Clone)]
pub struct Fixed;

impl Fixed {
    pub fn new() -> Self {
        Fixed
    }
}

impl Optimizer for Fixed {
    type State = ();

    fn new_state(&self, shape: &[usize]) -> Self::State {
        drop(shape);
        ()
    }

    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray) {
        #![allow(unused_variables)]
    }

    fn create_graph(&self) -> bool {
        false
    }
}
