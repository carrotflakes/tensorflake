mod backward;
mod computed;
mod function_call;
pub mod graph;
mod optimize;
mod optimizer;
pub mod param;

pub use backward::{chain, Backward};
pub use computed::Computed;
pub use function_call::FunctionCall;
pub use graph::gradients;
pub use optimize::{optimize, GradientsAccumulator};
pub use optimizer::Optimizer;
pub use param::Param;

pub fn backprop<T: Send + Sync + 'static>(x: T) -> Computed<T> {
    let y = Computed::new(x);
    chain(&[], &[y.clone()], true, "backprop", |_, _, _| vec![]);
    y
}
