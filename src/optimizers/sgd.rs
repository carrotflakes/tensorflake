use std::ops::Mul;

use crate::*;

#[derive(Clone)]
pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer<NDArray> for SGD {
    type State = ();

    fn new_state(&self, shape: &[usize]) -> Self::State {
        let _ = shape;
        ()
    }

    fn update(&mut self, data: &mut NDArray, state: &mut Self::State, grad: &NDArray) {
        let _ = state;
        *data = &*data + grad.mul(scalar(-self.learning_rate));
    }
}

#[test]
fn test() {
    super::test_optimizer(SGD::new(0.01));
}
