use std::rc::{Rc, Weak};

use crate::{Backward, Variable, VariableInner};

pub struct Funcall {
    pub(crate) function: Box<dyn Backward>,
    pub(crate) xs: Vec<Variable<true>>,
    pub(crate) ys: Vec<Weak<VariableInner>>,
    pub(crate) generation: u32,
}

impl Funcall {
    pub fn new(
        function: Box<dyn Backward>,
        xs: Vec<Variable<true>>,
        ys: &Vec<Variable<true>>,
        generation: u32,
    ) -> Self {
        Self {
            function,
            xs,
            ys: ys.iter().map(|y| Rc::downgrade(&y.inner)).collect(),
            generation,
        }
    }

    pub(crate) fn backward(&self, retain_grad: bool, enable_backprop: bool) {
        let ys = self.get_ys();
        let gys = ys
            .iter()
            .map(|y| y.get_grad().expect("ensure terminal variable's grad"))
            .collect();

        let gxs = self.function.backward(&self.xs, &ys, &gys, enable_backprop);

        for (x, gx) in self.xs.iter().zip(gxs) {
            x.add_grad(gx);
        }

        if !retain_grad {
            for y in &ys {
                y.clear_grad();
            }
        }
    }

    pub fn get_ys<const ENABLE_BACKPROP: bool>(&self) -> Vec<Variable<ENABLE_BACKPROP>> {
        self.ys
            .iter()
            .map(|y| Variable {
                inner: y.upgrade().unwrap(),
            })
            .collect()
    }
}
