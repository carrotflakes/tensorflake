use std::rc::{Rc, Weak};

use crate::{Backward, Variable, VariableInner};

pub struct Funcall {
    pub(crate) function: Box<dyn Backward>,
    pub(crate) xs: Vec<Variable<true>>,
    pub(crate) ys: Vec<Weak<VariableInner>>,
}

impl Funcall {
    pub fn new(
        function: Box<dyn Backward>,
        xs: Vec<Variable<true>>,
        ys: &Vec<Variable<true>>,
    ) -> Self {
        Self {
            function,
            xs,
            ys: ys.iter().map(|y| Rc::downgrade(&y.inner)).collect(),
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

pub fn sort_for_backward(mut fcs: Vec<Rc<Funcall>>) -> Vec<Rc<Funcall>> {
    let mut sorted = Vec::with_capacity(fcs.len());
    let ys = fcs.iter().flat_map(|fc| fc.get_ys()).collect::<Vec<_>>();
    let mut visited: Vec<_> = fcs
        .iter()
        .flat_map(|fc| &fc.xs)
        .filter(|v| !ys.contains(v))
        .cloned()
        .collect();
    while !fcs.is_empty() {
        let (a, b): (Vec<_>, _) = fcs
            .into_iter()
            .partition(|fc| fc.xs.iter().all(|x| visited.contains(&x)));
        if a.is_empty() {
            panic!("cycle detected");
        }
        visited.extend(a.iter().flat_map(|fc| fc.get_ys()));
        sorted.extend(a);
        fcs = b;
    }
    sorted.reverse();
    sorted
}
