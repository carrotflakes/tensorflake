use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use super::{Computed, FunctionCall};

pub trait Backward<T>: Sync + Send + 'static {
    fn backward(
        &self,
        xs: &Vec<Computed<T>>,
        ys: &Vec<Computed<T>>,
        gys: &Vec<Computed<T>>,
    ) -> Vec<Computed<T>>;

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

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

struct FnBackward<
    T: Sync + Send + 'static,
    F: Fn(&Vec<Computed<T>>, &Vec<Computed<T>>, &Vec<Computed<T>>) -> Vec<Computed<T>>
        + Sync
        + Send
        + 'static,
> {
    f: F,
    name: &'static str,
    _t: PhantomData<T>,
}

impl<
        T: Sync + Send + 'static,
        F: Fn(&Vec<Computed<T>>, &Vec<Computed<T>>, &Vec<Computed<T>>) -> Vec<Computed<T>>
            + Sync
            + Send
            + 'static,
    > Backward<T> for FnBackward<T, F>
{
    fn backward(
        &self,
        xs: &Vec<Computed<T>>,
        ys: &Vec<Computed<T>>,
        gys: &Vec<Computed<T>>,
    ) -> Vec<Computed<T>> {
        (self.f)(xs, ys, gys)
    }

    fn get_function_name(&self) -> Cow<'static, str> {
        self.name.into()
    }
}

pub fn chain<T: Sync + Send + 'static>(
    xs: &[Computed<T>],
    ys: &[Computed<T>],
    force_create_graph: bool,
    name: &'static str,
    backward: impl Fn(&Vec<Computed<T>>, &Vec<Computed<T>>, &Vec<Computed<T>>) -> Vec<Computed<T>>
        + Sync
        + Send
        + 'static,
) {
    if force_create_graph || xs.iter().any(|x| x.has_creator()) {
        let backward = Box::new(FnBackward {
            f: backward,
            name,
            _t: Default::default(),
        });
        let fc = FunctionCall::new(backward, xs.to_vec(), &ys);
        let fc = Arc::new(fc);
        for y in ys {
            y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
        }
    }
}
