mod layer;
mod linear;
mod mlp;
mod relu;

pub use layer::*;
pub use linear::*;
pub use mlp::*;
pub use relu::*;

use crate::{functions::*, *};

pub fn linear_simple(
    x: Variable,
    w: Variable,
    b: Variable,
) -> Variable {
    // NOTE: w*xの結果は捨てることができるが、そのためのAPIを用意していない
    call!(Add, call!(Matmul, w, x), b)
}

pub fn sigmoid_simple(x: Variable) -> Variable {
    call!(
        Div,
        Variable::new(scalar(1.0)),
        call!(Add, Variable::new(scalar(1.0)), call!(Exp, call!(Neg, x)))
    )
}
