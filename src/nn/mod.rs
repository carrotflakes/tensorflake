mod cnn;
mod dropout;
mod layer;
mod linear;
mod mlp;
mod relu;
mod select_net;
mod sigmoid;
mod softmax;

pub use cnn::*;
pub use dropout::*;
pub use layer::*;
pub use linear::*;
pub use mlp::*;
pub use relu::*;
pub use select_net::*;
pub use sigmoid::*;
pub use softmax::*;

use crate::{functions::*, *};

pub fn naive_linear(x: Variable, w: Variable, b: Variable) -> Variable {
    // NOTE: w*xの結果は捨てることができるが、そのためのAPIを用意していない
    call!(Add, call!(Matmul, w, x), b)
}

pub fn naive_sigmoid(x: Variable) -> Variable {
    call!(
        Div,
        Variable::new(scalar(1.0)),
        call!(Add, Variable::new(scalar(1.0)), call!(Exp, call!(Neg, x)))
    )
}
