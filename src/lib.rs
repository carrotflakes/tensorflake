mod execution_context;
pub mod export_dot;
mod function;
mod function_call;
pub mod functions;
pub mod graph;
mod impl_ops_for_tensor;
pub mod losses;
pub mod metrics;
pub mod ndarray_util;
pub mod nn;
mod optimize;
pub mod optimizers;
mod param;
pub mod param_bin;
mod tensor;

#[cfg(test)]
mod test;

pub use execution_context::*;
pub use function::*;
pub use function_call::FunctionCall;
pub use graph::gradients;
pub use metrics::{Metric, Metrics};
pub use ndarray_util::{scalar, IntoNDArray, NDArray};
pub use nn::Layer;
pub use optimize::*;
pub use optimizers::Optimizer;
pub use param::Param;
pub use tensor::Tensor;

pub type DefaultRng = rand_isaac::Isaac64Rng;

pub fn backprop(x: NDArray) -> Tensor {
    let y = Tensor::new(x);
    chain(&[], &[y.clone()], true, "backprop", |_, _, _| vec![]);
    y
}

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.to_owned()),*]).pop().unwrap()
    };
}
