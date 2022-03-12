use std::rc::Rc;

use crate::{Funcall, Tensor, Variable};

pub trait Function {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor>;
    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>;

    fn call<const ENABLE_BACKPROP: bool>(
        self,
        xs: Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>
    where
        Self: Sized + 'static,
    {
        let ys = self.forward(&xs);
        if !ENABLE_BACKPROP {
            ys.into_iter().map(|x| Variable::new(x)).collect()
        } else {
            let xs = unsafe { std::mem::transmute(xs) };
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
    fn backward(
        &self,
        xs: &Vec<Variable<true>>,
        gys: &Vec<Variable<true>>,
        enable_backprop: bool,
    ) -> Vec<Variable<true>>;
}

impl<T: Function> Backward for T {
    fn backward(
        &self,
        xs: &Vec<Variable<true>>,
        gys: &Vec<Variable<true>>,
        enable_backprop: bool,
    ) -> Vec<Variable<true>> {
        if enable_backprop {
            self.backward(xs, gys)
        } else {
            let xs = unsafe { std::mem::transmute(xs) };
            let gys = unsafe { std::mem::transmute(gys) };
            let gxs = self.backward::<false>(xs, gys);
            unsafe { std::mem::transmute(gxs) }
        }
    }
}
