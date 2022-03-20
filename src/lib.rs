pub mod export_dot;
mod funcall;
mod function;
pub mod functions;
pub mod graph;
pub mod losses;
pub mod nn;
mod optimization;
mod optimizees;
mod tensor;
pub mod tensor_util;
mod variable;

#[cfg(test)]
mod test;

pub use funcall::*;
pub use function::*;
pub use graph::gradients;
pub use nn::Layer;
pub use optimization::*;
pub use optimizees::*;
pub use tensor::*;
pub use variable::*;

pub fn backprop(x: Tensor) -> Variable {
    functions::CreateGraph::new(x).call(vec![]).pop().unwrap()
}

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.to_owned()),*]).pop().unwrap()
    };
}

pub fn call<const N: usize>(func: impl Function, xs: [Variable; N]) -> Variable {
    func.call(xs.to_vec()).pop().unwrap()
}
