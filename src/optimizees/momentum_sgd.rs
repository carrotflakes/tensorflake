use std::ops::Mul;

use crate::*;

pub struct MomentumSGDOptimizee {
    tensor: Tensor,
    momentum: f32,
    velocity: NDArray,
}

impl MomentumSGDOptimizee {
    pub fn new(ndarray: NDArray, momentum: f32) -> Param {
        Param::new(MomentumSGDOptimizee {
            velocity: NDArray::zeros(ndarray.shape()),
            tensor: Tensor::new(ndarray),
            momentum,
        })
    }
}

impl OptimizeeT for MomentumSGDOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.tensor.cut_chain();
        self.velocity *= self.momentum;
        self.velocity += &grad.mul(scalar(-lr));
        self.tensor = &self.tensor + &self.velocity.clone().into();
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| MomentumSGDOptimizee::new(tensor, 0.9), 0.01);
}
