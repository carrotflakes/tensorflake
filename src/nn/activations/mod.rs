mod relu;
mod sigmoid;
mod softmax;

pub use crate::functions::Tanh;
pub use relu::Relu;
pub use sigmoid::{naive_sigmoid, Sigmoid};
pub use softmax::Softmax;
