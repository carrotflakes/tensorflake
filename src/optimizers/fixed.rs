use crate::*;

#[derive(Clone)]
pub struct Fixed;

impl Fixed {
    pub fn new() -> Self {
        Fixed
    }
}

impl<T: Send + Sync + 'static> Optimizer<T> for Fixed {
    type State = ();

    fn new_state(&self, shape: &[usize]) -> Self::State {
        let _ = shape;
        ()
    }

    fn update(&mut self, data: &mut T, state: &mut Self::State, grad: &T) {
        #![allow(unused_variables)]
    }
}
