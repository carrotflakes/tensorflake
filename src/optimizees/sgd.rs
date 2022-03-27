use std::ops::Mul;

use crate::*;

pub struct SGDOptimizee {
    tensor: Tensor,
}

impl SGDOptimizee {
    pub fn new(ndarray: NDArray) -> Param {
        Param::new(SGDOptimizee {
            tensor: Tensor::new(ndarray),
        })
    }
}

impl OptimizeeT for SGDOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.tensor.cut_chain();
        self.tensor = &self.tensor + &grad.mul(scalar(-lr)).into();
    }
}

#[test]
fn test() {
    super::test_optimizee(SGDOptimizee::new, 0.01);
}
