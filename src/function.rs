use std::rc::Rc;

use crate::{Funcall, Tensor, Variable};

pub trait Function {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor>;
    fn backward(&self, xs: &Vec<Variable<true>>, gys: &Vec<Variable<true>>) -> Vec<Variable<true>>;

    fn call<const ENABLE_BACKPROP: bool>(
        self,
        xs: Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>
    where
        Self: Sized + 'static,
    {
        if !ENABLE_BACKPROP {
            self.forward(&xs)
                .into_iter()
                .map(|x| Variable::new(x))
                .collect()
        } else {
            let ys = self.forward(&xs);
            let xs = unsafe { std::mem::transmute::<_, Vec<Variable<true>>>(xs) };
            let fc = Funcall::new(Box::new(self), xs, ys);

            let fc = Rc::new(fc);
            for y in &fc.output {
                y.inner.creator.replace(Some(fc.clone()));
            }
            unsafe { std::mem::transmute(fc.output.clone()) }
        }
    }
}

pub trait Backward {
    fn backward(&self, xs: &Vec<Variable<true>>, gys: &Vec<Variable<true>>) -> Vec<Variable<true>>;
}

impl<T: Function> Backward for T {
    fn backward(&self, xs: &Vec<Variable<true>>, gys: &Vec<Variable<true>>) -> Vec<Variable<true>> {
        self.backward(xs, gys)
    }
}
