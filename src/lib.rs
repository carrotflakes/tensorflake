pub mod export_dot;
mod funcall;
mod function;
pub mod functions;
pub mod graph;
mod impl_ops_for_variable;
pub mod losses;
pub mod nn;
mod optimization;
mod optimizees;
// pub mod param_bin;
pub mod ndarray_util;
mod tensor;

#[cfg(test)]
mod test;

pub use funcall::*;
pub use function::*;
pub use graph::gradients;
pub use ndarray_util::{scalar, IntoNDArray, NDArray};
pub use nn::Layer;
pub use optimization::*;
pub use optimizees::*;
pub use tensor::*;

pub fn backprop(x: NDArray) -> Tensor {
    functions::CreateGraph::new(x).call(vec![]).pop().unwrap()
}

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.to_owned()),*]).pop().unwrap()
    };
}

pub fn call<const N: usize>(func: impl Function, xs: [Tensor; N]) -> Tensor {
    func.call(xs.to_vec()).pop().unwrap()
}
