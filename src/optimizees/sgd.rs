use std::ops::Mul;

use crate::*;

pub struct SGDOptimizee {
    tensor: Tensor,
}

impl SGDOptimizee {
    pub fn new(tensor: Tensor) -> Optimizee {
        Optimizee::new(SGDOptimizee { tensor })
    }
}

impl OptimizeeT for SGDOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn update(&mut self, grad: &Tensor, lr: f32) {
        self.tensor += &grad.mul(scalar(-lr));
    }
}

#[test]
fn test() {
    super::test_optimizee(SGDOptimizee::new, 0.01);
}
