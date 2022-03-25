use std::ops::Mul;

use crate::*;

pub struct Fixed {
    tensor: NDArray,
}

impl Fixed {
    pub fn new(tensor: NDArray) -> Optimizee {
        Optimizee::new(Fixed { tensor })
    }
}

impl OptimizeeT for Fixed {
    fn tensor_ref(&self) -> &NDArray {
        &self.tensor
    }

    fn set(&mut self, tensor: NDArray) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.tensor += &grad.mul(scalar(-lr));
    }

    fn create_graph(&self) -> bool {
        false
    }
}
