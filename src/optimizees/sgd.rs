use std::ops::{AddAssign, Mul};

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

    fn update(&self, grad: &Tensor, lr: f32) {
        #[allow(mutable_transmutes)]
        let tensor = unsafe { std::mem::transmute::<_, &mut Tensor>(&self.tensor) };
        tensor.add_assign(&grad.mul(scalar(-lr)));
    }
}

#[test]
fn test() {
    let px = SGDOptimizee::new(scalar(0.0));

    let mut last_loss = 1000.0;
    for _ in 0..10 {
        let x = px.get();
        let y = call!(functions::Add, x, x);
        let loss = call!(
            functions::Pow::new(2.0),
            call!(functions::Sub, y, Variable::new(scalar(6.0)))
        );
        assert!(loss[[]] < last_loss);
        last_loss = loss[[]];

        optimize(&loss, 0.01);
    }
}
