use std::{borrow::Cow, sync::Arc};

use super::{FunctionCall, Tensor};

pub trait Function: Sync + Send + 'static {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor>;
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor>;

    fn into_backward(self, xs: &Vec<Tensor>) -> Box<dyn Backward>
    where
        Self: Sized,
    {
        #![allow(unused_variables)]

        Box::new(self)
    }

    fn call(self, xs: Vec<Tensor>) -> Vec<Tensor>
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
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor>;

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
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        self.backward(xs, ys, gys)
    }
}

struct FnBackward<
    F: Fn(&Vec<Tensor>, &Vec<Tensor>, &Vec<Tensor>) -> Vec<Tensor> + Sync + Send + 'static,
> {
    f: F,
    name: &'static str,
}

impl<F: Fn(&Vec<Tensor>, &Vec<Tensor>, &Vec<Tensor>) -> Vec<Tensor> + Sync + Send + 'static>
    Backward for FnBackward<F>
{
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        (self.f)(xs, ys, gys)
    }

    fn get_function_name(&self) -> Cow<'static, str> {
        self.name.into()
    }
}

pub fn chain(
    xs: &[Tensor],
    ys: &[Tensor],
    force_create_graph: bool,
    name: &'static str,
    backward: impl Fn(&Vec<Tensor>, &Vec<Tensor>, &Vec<Tensor>) -> Vec<Tensor> + Sync + Send + 'static,
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
