use std::sync::{Arc, Weak};

use super::{computed::ComputedInner, Backward, Computed};

pub struct FunctionCall<T> {
    pub(crate) backward: Box<dyn Backward<T>>,
    pub(crate) xs: Vec<Computed<T>>,
    pub(crate) ys: Vec<Weak<ComputedInner<T>>>,
}

impl<T> FunctionCall<T> {
    pub fn new(backward: Box<dyn Backward<T>>, xs: Vec<Computed<T>>, ys: &[Computed<T>]) -> Self {
        Self {
            backward,
            xs,
            ys: ys.iter().map(|y| Arc::downgrade(&y.inner)).collect(),
        }
    }

    pub fn get_ys(&self) -> Vec<Computed<T>> {
        self.ys
            .iter()
            .map(|y| Computed {
                inner: y.upgrade().unwrap(),
            })
            .collect()
    }
}
