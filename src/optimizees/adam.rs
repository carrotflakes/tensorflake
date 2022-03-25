use crate::*;

const EPS: f32 = 1e-8;

pub struct AdamOptimizee {
    tensor: NDArray,
    beta1: f32,
    beta2: f32,
    mom: NDArray, // TODO: owned mom and vel
    vel: NDArray,
}

impl AdamOptimizee {
    pub fn new(tensor: NDArray) -> Param {
        Param::new(AdamOptimizee {
            mom: NDArray::zeros(tensor.shape()),
            vel: NDArray::zeros(tensor.shape()),
            tensor,
            beta1: 0.9,
            beta2: 0.999,
        })
    }

    pub fn new_with_params(tensor: NDArray, beta1: f32, beta2: f32) -> Param {
        Param::new(AdamOptimizee {
            mom: NDArray::zeros(tensor.shape()),
            vel: NDArray::zeros(tensor.shape()),
            tensor,
            beta1,
            beta2,
        })
    }
}

impl OptimizeeT for AdamOptimizee {
    fn tensor_ref(&self) -> &NDArray {
        &self.tensor
    }
    
    fn set(&mut self, tensor: NDArray) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.mom = (&self.mom * self.beta1 + grad * (1.0 - self.beta1)).into_ndarray();
        self.vel =
            (&self.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_ndarray();
        self.tensor += &(&self.mom / self.vel.map(|x| x.sqrt() + EPS) * -lr);
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| AdamOptimizee::new(tensor), 0.01);
}
