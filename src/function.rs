use std::sync::Arc;

use crate::{Funcall, Variable};

pub trait Function: 'static {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable>;
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable>;

    fn into_backward(self, xs: &Vec<Variable>) -> Box<dyn Backward>
    where
        Self: Sized,
    {
        #![allow(unused_variables)]

        Box::new(self)
    }

    fn call(self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized,
    {
        let ys = self.forward(&xs);
        if Self::IS_FORCE_CREATE_GRAPH || xs.iter().any(|x| x.has_creator()) {
            let backward = self.into_backward(&xs);
            let fc = Funcall::new(backward, xs, &ys);
            let fc = Arc::new(fc);
            for y in &ys {
                y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
            }
        }
        ys
    }

    const IS_FORCE_CREATE_GRAPH: bool = false;
}

pub trait Backward {
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable>;

    fn get_function_name(&self) -> &'static str {
        let name = std::any::type_name::<Self>();
        if name.starts_with("tensorflake::functions::") {
            name.split("::").last().unwrap()
        } else if name.starts_with("tensorflake::") {
            &name["tensorflake::".len()..]
        } else {
            name
        }
    }

    fn get_optimizee(&self) -> Option<crate::Optimizee> {
        None
    }
}

impl<T: Function> Backward for T {
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        self.backward(xs, ys, gys)
    }
}

struct FnBackward<F: Fn(&Vec<Variable>, &Vec<Variable>, &Vec<Variable>) -> Vec<Variable> + 'static>
{
    f: F,
    name: &'static str,
}

impl<F: Fn(&Vec<Variable>, &Vec<Variable>, &Vec<Variable>) -> Vec<Variable> + 'static> Backward
    for FnBackward<F>
{
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        (self.f)(xs, ys, gys)
    }

    fn get_function_name(&self) -> &'static str {
        self.name
    }
}

pub fn chain(
    xs: &[Variable],
    ys: &[Variable],
    force_create_graph: bool,
    name: &'static str,
    backward: impl Fn(&Vec<Variable>, &Vec<Variable>, &Vec<Variable>) -> Vec<Variable> + 'static,
) {
    if force_create_graph || xs.iter().any(|x| x.has_creator()) {
        let backward = Box::new(FnBackward { f: backward, name });
        let fc = Funcall::new(backward, xs.to_vec(), &ys);
        let fc = Arc::new(fc);
        for y in ys {
            y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
        }
    }
}
