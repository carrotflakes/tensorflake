pub mod contrib;
pub mod core;
pub mod export_dot;
pub mod functions;
mod impl_ops_for_tensor;
pub mod initializers;
pub mod losses;
pub mod metrics;
pub mod ndarray_util;
pub mod nn;
pub mod optimizers;
pub mod param_bin;
pub mod regularizers;
pub mod training;

#[cfg(test)]
mod test;

pub use crate::core::*;
pub use metrics::{Metric, Metrics};
pub use ndarray_util::{scalar, IntoNDArray, NDArray};
pub use nn::Layer;

pub type DefaultRng = rand_isaac::Isaac64Rng;

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.to_owned()),*]).pop().unwrap()
    };
}
