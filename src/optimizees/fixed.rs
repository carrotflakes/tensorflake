use std::ops::Mul;

use crate::*;

pub struct Fixed {
    tensor: Tensor,
}

impl Fixed {
    pub fn new(ndarray: NDArray) -> Param {
        Param::new(Fixed {
            tensor: Tensor::new(ndarray),
        })
    }
}

impl OptimizeeT for Fixed {
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

    fn create_graph(&self) -> bool {
        false
    }
}
