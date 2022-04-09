mod relu;
mod sigmoid;
mod softmax;

pub use crate::functions::tanh;
pub use relu::{relu, Relu};
pub use sigmoid::{naive_sigmoid, sigmoid, Sigmoid};
pub use softmax::{softmax, Softmax};
