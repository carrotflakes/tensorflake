mod cnn;
mod dropout;
mod layer;
mod linear;
mod mlp;
pub mod regularizers;
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

pub fn naive_sigmoid(x: Tensor) -> Tensor {
    call!(
        Div,
        Tensor::new(scalar(1.0)),
        call!(Add, Tensor::new(scalar(1.0)), call!(Exp, call!(Neg, x)))
    )
}
