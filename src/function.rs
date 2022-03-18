use std::{any::TypeId, sync::Arc};

use crate::{Funcall, Tensor, Variable, functions::CreateGraph};

pub trait Function: 'static {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor>;
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable>;

    fn into_backward(self, xs: &Vec<Variable>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        #![allow(unused_variables)]

        Box::new(self)
    }

    // fn call(self, xs: Vec<Variable>) -> Vec<Variable>
    // where
    //     Self: Sized + 'static,
    // {
    //     let ys = self.forward(&xs);
    //     let ys: Vec<_> = ys.into_iter().map(|y| Variable::new(y)).collect();

    //     let recorders = xs
    //         .iter()
    //         .flat_map(|x| {
    //             x.inner
    //                 .attrs
    //                 .lock()
    //                 .unwrap()
    //                 .recorder
    //                 .clone()
    //                 .and_then(|r| if r.is_prevented() { None } else { Some(r) })
    //         })
    //         .collect::<Vec<_>>();

    //     if !recorders.is_empty() {
    //         let recorder = Recorder::merge(recorders);
    //         let backward = self.into_backward(&xs);
    //         let fc = Funcall::new(backward, xs, ys.clone());
    //         recorder.push(fc);

    //         for y in &ys {
    //             y.set_recorder(recorder.clone());
    //         }
    //     }

    //     ys
    // }
    
    fn call(self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        let ys = self.forward(&xs);
        let ys: Vec<_> = ys.into_iter().map(|y| Variable::new(y)).collect();
        if self.is_force_create_graph() || xs.iter().any(|x| x.has_creator()) {
            let backward = self.into_backward(&xs);
            let fc = Funcall::new(backward, xs, &ys);
            let fc = Arc::new(fc);
            for y in &ys {
                y.inner.attrs.lock().unwrap().creator = Some(fc.clone());
            }
        }
        ys
    }

    fn is_force_create_graph(&self) -> bool {
        TypeId::of::<Self>() == TypeId::of::<CreateGraph>()
    }
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
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        self.backward(xs, ys, gys)
    }
}
