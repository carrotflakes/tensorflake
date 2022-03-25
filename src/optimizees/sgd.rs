use std::ops::Mul;

use crate::*;

pub struct SGDOptimizee {
    tensor: NDArray,
}

impl SGDOptimizee {
    pub fn new(tensor: NDArray) -> Param {
        Param::new(SGDOptimizee { tensor })
    }
}

impl OptimizeeT for SGDOptimizee {
    fn tensor_ref(&self) -> &NDArray {
        &self.tensor
    }
    
    fn set(&mut self, tensor: NDArray) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.tensor += &grad.mul(scalar(-lr));
    }
}

#[test]
fn test() {
    super::test_optimizee(SGDOptimizee::new, 0.01);
}
