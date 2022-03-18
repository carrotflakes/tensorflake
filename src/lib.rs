pub mod export_dot;
mod funcall;
mod function;
pub mod functions;
pub mod nn;
// mod recorder;
mod tensor;
mod variable;

#[cfg(test)]
mod test;

pub use funcall::*;
pub use function::*;
// pub use recorder::*;
pub use tensor::*;
pub use variable::*;

pub fn backprop(x: Tensor) -> Variable {
    functions::CreateGraph::new(x).call(vec![]).pop().unwrap()
}

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.clone()),*]).pop().unwrap()
    };
}
