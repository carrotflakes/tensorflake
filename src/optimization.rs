use std::sync::{Arc, Mutex};

use crate::*;

pub trait OptimizeeT: 'static {
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
}

pub fn optimize(loss: &Tensor, lr: f32) {
    let funcalles = graph::collect_funcalls(vec![loss.clone()]);
    let mut params = Vec::new();
    let mut trainables = Vec::new();
    for fc in funcalles {
        if let Some(o) = fc.backward.get_param() {
            params.push(o);
            trainables.push(fc.get_ys()[0].clone());
        }
    }
    let grads = gradients(&vec![loss.clone()], &trainables, false);
    for (param, grad) in params.iter().zip(grads.iter()) {
        param.update(grad, lr);
    }
}
