mod momentum_sgd;
mod sgd;

pub use momentum_sgd::*;
pub use sgd::*;

#[cfg(test)]
fn test_optimizee(f: impl Fn(crate::Tensor) -> crate::Optimizee) {
    use crate::*;

    let px = f(scalar(0.0));

    let loss_fn = || {
        let x = px.get();
        let y = call!(functions::Add, x, x);
        let loss = call!(
            functions::Pow::new(2.0),
            call!(functions::Sub, y, Variable::new(scalar(6.0)))
        );
        loss
    };

    let first_loss = loss_fn()[[]];

    for _ in 0..100 {
        let loss = loss_fn();

        optimize(&loss, 0.01);
    }

    let last_loss = loss_fn()[[]];
    assert!(last_loss < first_loss * 0.01);
}
