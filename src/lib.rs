pub mod contrib;
pub mod core;
pub mod functions;
mod impl_ops_for_tensor;
pub mod initializers;
pub mod losses;
pub mod metrics;
pub mod ndarray_util;
pub mod nn;
pub mod optimizers;
pub mod regularizers;
pub mod training;

#[cfg(test)]
mod test;

pub use crate::core::*;
pub use contrib::{export_dot, param_bin};
pub use metrics::{Metric, Metrics};
pub use ndarray_util::{scalar, IntoNDArray, NDArray};
pub use nn::Layer;

pub use ndarray;
pub use ndarray_rand;

pub type DefaultRng = rand_isaac::Isaac64Rng;
