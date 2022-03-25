use std::ops::Mul;

use crate::*;

pub struct MomentumSGDOptimizee {
    tensor: NDArray,
    momentum: f32,
    velocity: NDArray,
}

impl MomentumSGDOptimizee {
    pub fn new(tensor: NDArray, momentum: f32) -> Param {
        Param::new(MomentumSGDOptimizee {
            velocity: NDArray::zeros(tensor.shape()),
            tensor,
            momentum,
        })
    }
}

impl OptimizeeT for MomentumSGDOptimizee {
    fn tensor_ref(&self) -> &NDArray {
        &self.tensor
    }
    
    fn set(&mut self, tensor: NDArray) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.velocity *= self.momentum;
        self.velocity += &grad.mul(scalar(-lr));
        self.tensor += &self.velocity;
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| MomentumSGDOptimizee::new(tensor, 0.9), 0.01);
}
