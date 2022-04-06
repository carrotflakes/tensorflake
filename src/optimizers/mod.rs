mod adam;
mod adamw;
mod fixed;
mod momentum_sgd;
mod sgd;
mod with_regularization;

pub use adam::AdamOptimizer;
pub use adamw::AdamWOptimizer;
pub use fixed::Fixed;
pub use momentum_sgd::MomentumSGDOptimizer;
pub use sgd::SGDOptimizer;
pub use with_regularization::WithRegularization;

use crate::{NDArray, Tensor};

pub trait Optimizer: Clone + Sync + Send + 'static {
    type State: Sync + Send + 'static;

    fn new_state(&self, shape: &[usize]) -> Self::State;
    fn update(&mut self, tensor: &mut Tensor, state: &mut Self::State, grad: &NDArray);

    fn create_graph(&self) -> bool {
        true
    }
}

#[cfg(test)]
fn test_optimizer(optimizer: impl Optimizer) {
    use crate::*;

    let px = crate::Param::new(scalar(0.0), optimizer);

    let loss_fn = || {
        let x = px.get_tensor();
        let y = x.clone() + x;
        (y - Tensor::new(scalar(6.0))).pow(2.0)
    };

    let first_loss = loss_fn()[[]];

    for _ in 0..100 {
        let loss = loss_fn();
        optimize(&loss);
    }

    let last_loss = loss_fn()[[]];
    println!("loss: {} -> {}", first_loss, last_loss);
    assert!(last_loss < first_loss * 0.01);
}
