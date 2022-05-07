mod backward;
mod computed;
mod function_call;
pub mod graph;
mod optimize;
mod optimizer;
pub mod param;

pub use backward::{chain, Backward};
pub use function_call::FunctionCall;
pub use graph::gradients;
pub use optimize::{optimize, GradientsAccumulator};
pub use optimizer::Optimizer;
pub use param::Param;

use crate::NDArray;

pub type Computed = computed::Computed<NDArray>;

impl Into<Computed> for NDArray {
    fn into(self) -> Computed {
        Computed::new(self)
    }
}

pub fn backprop(x: NDArray) -> Computed {
    let y = Computed::new(x);
    chain(&[], &[y.clone()], true, "backprop", |_, _, _| vec![]);
    y
}
