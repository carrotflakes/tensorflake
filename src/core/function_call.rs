use std::sync::{Arc, Weak};

use super::{tensor::ComputedInner, Backward, Computed};

pub struct FunctionCall {
    pub(crate) backward: Box<dyn Backward>,
    pub(crate) xs: Vec<Computed>,
    pub(crate) ys: Vec<Weak<ComputedInner>>,
}

impl FunctionCall {
    pub fn new(backward: Box<dyn Backward>, xs: Vec<Computed>, ys: &[Computed]) -> Self {
        Self {
            backward,
            xs,
            ys: ys.iter().map(|y| Arc::downgrade(&y.inner)).collect(),
        }
    }

    pub fn get_ys(&self) -> Vec<Computed> {
        self.ys
            .iter()
            .map(|y| Computed {
                inner: y.upgrade().unwrap(),
            })
            .collect()
    }
}
