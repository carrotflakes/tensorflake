use crate::*;

const EPS: f32 = 1e-8;

pub struct AdamOptimizee {
    tensor: Tensor,
    beta1: f32,
    beta2: f32,
    mom: NDArray, // TODO: owned mom and vel
    vel: NDArray,
}

impl AdamOptimizee {
    pub fn new(ndarray: NDArray) -> Param {
        Param::new(AdamOptimizee {
            mom: NDArray::zeros(ndarray.shape()),
            vel: NDArray::zeros(ndarray.shape()),
            tensor: Tensor::new(ndarray),
            beta1: 0.9,
            beta2: 0.999,
        })
    }

    pub fn new_with_params(ndarray: NDArray, beta1: f32, beta2: f32) -> Param {
        Param::new(AdamOptimizee {
            mom: NDArray::zeros(ndarray.shape()),
            vel: NDArray::zeros(ndarray.shape()),
            tensor: Tensor::new(ndarray),
            beta1,
            beta2,
        })
    }
}

impl OptimizeeT for AdamOptimizee {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray, lr: f32) {
        self.tensor.cut_chain();
        self.mom = (&self.mom * self.beta1 + grad * (1.0 - self.beta1)).into_ndarray();
        self.vel =
            (&self.vel * self.beta2 + grad.map(|x| x.powi(2)) * (1.0 - self.beta2)).into_ndarray();
        self.tensor = &self.tensor
            + &(&self.mom / self.vel.map(|x| x.sqrt() + EPS) * -lr)
                .into_ndarray()
                .into();
    }
}

#[test]
fn test() {
    super::test_optimizee(|tensor| AdamOptimizee::new(tensor), 0.01);
}
