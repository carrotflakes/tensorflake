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

use crate::NDArray;

pub type ComputedNDA = Computed<NDArray>;
pub type ParamNDA = param::Param<NDArray>;

impl Into<ComputedNDA> for NDArray {
    fn into(self) -> ComputedNDA {
        ComputedNDA::new(self)
    }
}

pub fn backprop<T: Send + Sync + 'static>(x: T) -> Computed<T> {
    let y = Computed::new(x);
    chain(&[], &[y.clone()], true, "backprop", |_, _, _| vec![]);
    y
}

impl graph::One for super::NDArray {
    fn clone_filled_ones(&self) -> Self {
        super::NDArray::ones(self.shape())
    }

    fn shape(&self) -> &[usize] {
        super::NDArray::shape(self)
    }
}
