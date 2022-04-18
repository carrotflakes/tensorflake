use std::{borrow::Cow, sync::Arc};

use super::{FunctionCall, Computed};

pub trait Function: Sync + Send + 'static {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed>;
    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed>;

    fn into_backward(self, xs: &Vec<Computed>) -> Box<dyn Backward>
    where
        Self: Sized,
    {
        #![allow(unused_variables)]

        Box::new(self)
    }

    fn call(self, xs: Vec<Computed>) -> Vec<Computed>
    where
        Self: Sized,
    {
        let ys = self.forward(&xs);
        if xs.iter().any(|x| x.has_creator()) {
            let backward = self.into_backward(&xs);
            let fc = FunctionCall::new(backward, xs, &ys);
            let fc = Arc::new(fc);
            for y in &ys {
                y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
            }
        }
        ys
    }
}

pub trait Backward: Sync + Send {
    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed>;

    fn get_function_name(&self) -> Cow<'static, str> {
        let name = std::any::type_name::<Self>();
        if name.starts_with("tensorflake::functions::") {
            name.split("::").last().unwrap()
        } else if name.starts_with("tensorflake::") {
            &name["tensorflake::".len()..]
        } else {
            name
        }
        .into()
    }

    fn get_param(&self) -> Option<crate::Param> {
        None
    }
}

impl<T: Function> Backward for T {
    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        self.backward(xs, ys, gys)
    }
}

struct FnBackward<
    F: Fn(&Vec<Computed>, &Vec<Computed>, &Vec<Computed>) -> Vec<Computed> + Sync + Send + 'static,
> {
    f: F,
    name: &'static str,
}

impl<F: Fn(&Vec<Computed>, &Vec<Computed>, &Vec<Computed>) -> Vec<Computed> + Sync + Send + 'static>
    Backward for FnBackward<F>
{
    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        (self.f)(xs, ys, gys)
    }

    fn get_function_name(&self) -> Cow<'static, str> {
        self.name.into()
    }
}

pub fn chain(
    xs: &[Computed],
    ys: &[Computed],
    force_create_graph: bool,
    name: &'static str,
    backward: impl Fn(&Vec<Computed>, &Vec<Computed>, &Vec<Computed>) -> Vec<Computed> + Sync + Send + 'static,
) {
    if force_create_graph || xs.iter().any(|x| x.has_creator()) {
        let backward = Box::new(FnBackward { f: backward, name });
        let fc = FunctionCall::new(backward, xs.to_vec(), &ys);
        let fc = Arc::new(fc);
        for y in ys {
            y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
        }
    }
}
