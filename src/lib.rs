pub mod export_dot;
mod funcall;
mod function;
pub mod functions;
pub mod nn;
mod tensor;
mod variable;

#[cfg(test)]
mod test;

pub use funcall::*;
pub use function::*;
pub use tensor::*;
pub use variable::*;

pub fn backprop(x: Tensor) -> Variable {
    functions::CreateGraph::new(x).call(vec![]).pop().unwrap()
}

pub fn trainable(x: Tensor) -> Variable {
    let v = functions::CreateGraph::new(x).call(vec![]).pop().unwrap();
    v.set_trainable(true);
    v
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
