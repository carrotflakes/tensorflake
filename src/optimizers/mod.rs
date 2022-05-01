mod adam;
mod adamw;
mod fixed;
mod momentum_sgd;
mod sgd;
mod with_regularization;

pub use adam::Adam;
pub use adamw::AdamW;
pub use fixed::Fixed;
pub use momentum_sgd::MomentumSGD;
pub use sgd::SGD;
pub use with_regularization::WithRegularization;

#[cfg(test)]
fn test_optimizer(optimizer: impl crate::Optimizer + Clone) {
    use crate::*;

    let px = crate::Param::new(scalar(0.0), "param".into(), optimizer);

    let loss_fn = || {
        let x = px.get();
        let y = x.clone() + x;
        (y - Computed::new(scalar(6.0))).pow(2.0)
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
