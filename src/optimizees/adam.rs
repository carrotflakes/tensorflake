use crate::*;

const EPS: f32 = 1e-8;

pub struct AdamOptimizee {
    tensor: Tensor,
    beta1: f32,
    beta2: f32,
    mom: Tensor, // TODO: owned mom and vel
    vel: Tensor,
}

impl AdamOptimizee {
    pub fn new(tensor: Tensor) -> Optimizee {
        Optimizee::new(AdamOptimizee {
            mom: Tensor::zeros(tensor.shape()),
            vel: Tensor::zeros(tensor.shape()),
            tensor,
            beta1: 0.9,
            beta2: 0.999,
        })
    }

    pub fn new_with_params(tensor: Tensor, beta1: f32, beta2: f32) -> Optimizee {
        Optimizee::new(AdamOptimizee {
            mom: Tensor::zeros(tensor.shape()),
            vel: Tensor::zeros(tensor.shape()),
            tensor,
            beta1,
            beta2,
        })
    }
}

impl OptimizeeT for AdamOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn update(&mut self, grad: &Tensor, lr: f32) {
        self.mom = (&self.mom * self.beta1 + grad * (1.0 - self.beta1)).into_tensor();
        self.vel =
            (&self.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_tensor();
        self.tensor += &(&self.mom / self.vel.map(|x| x.sqrt() + EPS) * -lr);
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| AdamOptimizee::new(tensor), 0.01);
}
