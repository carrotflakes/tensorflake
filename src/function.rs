use std::rc::Rc;

use crate::{Funcall, Tensor, Variable};

pub trait Function {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor>;
    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable>;

    fn call(self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        let fc = Funcall::new(Box::new(self), xs);
        let fc = Rc::new(fc);
        for y in &fc.output {
            y.inner.creator.replace(Some(fc.clone()));
        }
        fc.output.clone()
    }
}
