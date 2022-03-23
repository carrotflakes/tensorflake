use std::sync::{Arc, Weak};

use crate::{Backward, Variable, VariableInner};

pub struct Funcall {
    pub(crate) backward: Box<dyn Backward>,
    pub(crate) xs: Vec<Variable>,
    pub(crate) ys: Vec<Weak<VariableInner>>,
}

impl Funcall {
    pub fn new(backward: Box<dyn Backward>, xs: Vec<Variable>, ys: &[Variable]) -> Self {
        Self {
            backward,
            xs,
            ys: ys.iter().map(|y| Arc::downgrade(&y.inner)).collect(),
        }
    }

    pub fn get_ys(&self) -> Vec<Variable> {
        self.ys
            .iter()
            .map(|y| Variable {
                inner: y.upgrade().unwrap(),
            })
            .collect()
    }
}
