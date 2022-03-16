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
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>;

    fn into_backward(self, xs: &Vec<Variable<true>>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        #![allow(unused_variables)]

        Box::new(self)
    }

    fn call<const ENABLE_BACKPROP: bool>(
        self,
        xs: Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>
    where
        Self: Sized + 'static,
    {
        let ys = self.forward(&xs);
        if !ENABLE_BACKPROP {
            ys.into_iter().map(|y| Variable::new(y)).collect()
        } else {
            let xs = unsafe { std::mem::transmute(xs) };
            let backward = self.into_backward(&xs);

            let ys: Vec<_> = ys.into_iter().map(|y| Variable::new(y)).collect();
            let fc = Funcall::new(backward, xs, &ys);
            let fc = Rc::new(fc);
            for y in &ys {
                y.inner.attrs.borrow_mut().creator = Some(fc.clone());
            }
            unsafe { std::mem::transmute(ys) }
        }
    }
}

pub trait Backward {
    fn backward(
        &self,
        xs: &Vec<Variable<true>>,
        ys: &Vec<Variable<true>>,
        gys: &Vec<Variable<true>>,
        enable_backprop: bool,
    ) -> Vec<Variable<true>>;

    fn get_function_name(&self) -> &'static str {
        let name = std::any::type_name::<Self>();
        if name.starts_with("ruzero::functions::") {
            name.split("::").last().unwrap()
        } else {
            name
        }
    }
}

impl<T: Function> Backward for T {
    fn backward(
        &self,
        xs: &Vec<Variable<true>>,
        ys: &Vec<Variable<true>>,
        gys: &Vec<Variable<true>>,
        enable_backprop: bool,
    ) -> Vec<Variable<true>> {
        if enable_backprop {
            self.backward(xs, ys, gys)
        } else {
            let xs = unsafe { std::mem::transmute(xs) };
            let ys = unsafe { std::mem::transmute(ys) };
            let gys = unsafe { std::mem::transmute(gys) };
            let gxs = self.backward::<false>(xs, ys, gys);
            unsafe { std::mem::transmute(gxs) }
        }
    }
}
