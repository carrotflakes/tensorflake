use std::sync::{Arc, Mutex};

use crate::*;

pub trait OptimizeeT: Sync + Send + 'static {
    fn tensor_ref(&self) -> &NDArray;
    fn set(&mut self, tensor: NDArray);
    fn update(&mut self, grad: &NDArray, lr: f32);

    fn create_graph(&self) -> bool {
        true
    }
}

pub struct Param {
    inner: Arc<Mutex<dyn OptimizeeT>>,
}

impl Param {
    pub fn new(inner: impl OptimizeeT) -> Param {
        Param {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn get_tensor(&self) -> Tensor {
        let inner = self.inner.lock().unwrap();
        let v = Tensor::new(inner.tensor_ref().clone());
        if inner.create_graph() {
            let creator = Funcall {
                backward: Box::new(Param {
                    inner: self.inner.clone(),
                }),
                xs: vec![],
                ys: vec![std::sync::Arc::downgrade(&v.inner)],
            };
            v.inner.attrs.lock().unwrap().creator = Some(std::sync::Arc::new(creator));
        }
        v
    }

    pub fn set(&mut self, ndarray: NDArray) {
        let mut inner = self.inner.lock().unwrap();
        inner.set(ndarray);
    }

    pub fn update(&self, grad: &Tensor, lr: f32) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(&grad, lr);
    }
}

impl Clone for Param {
    fn clone(&self) -> Param {
        Param {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Param {}

impl std::hash::Hash for Param {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl Backward for Param {
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]
        vec![]
    }

    fn get_param(&self) -> Option<Param> {
        Some(Param {
            inner: self.inner.clone(),
        })
    }

    fn get_function_name(&self) -> &'static str {
        "Param"
    }
}