use std::ops::Mul;

use crate::*;

pub struct MomentumSGDOptimizee {
    tensor: Tensor,
    momentum: f32,
    velocity: Tensor,
}

impl MomentumSGDOptimizee {
    pub fn new(tensor: Tensor, momentum: f32) -> Optimizee {
        Optimizee::new(MomentumSGDOptimizee {
            velocity: Tensor::zeros(tensor.shape()),
            tensor,
            momentum,
        })
    }
}

impl OptimizeeT for MomentumSGDOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn update(&mut self, grad: &Tensor, lr: f32) {
        self.velocity *= self.momentum;
        self.velocity += &grad.mul(scalar(-lr));
        self.tensor += &self.velocity;
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| MomentumSGDOptimizee::new(tensor, 0.9));
}
