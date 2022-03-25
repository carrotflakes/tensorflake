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

pub struct Optimizee {
    inner: Arc<Mutex<dyn OptimizeeT>>,
}

impl Optimizee {
    pub fn new(inner: impl OptimizeeT) -> Optimizee {
        Optimizee {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn get(&self) -> Tensor {
        let inner = self.inner.lock().unwrap();
        let v = Tensor::new(inner.tensor_ref().clone());
        if inner.create_graph() {
            let creator = Funcall {
                backward: Box::new(Optimizee {
                    inner: self.inner.clone(),
                }),
                xs: vec![],
                ys: vec![std::sync::Arc::downgrade(&v.inner)],
            };
            v.inner.attrs.lock().unwrap().creator = Some(std::sync::Arc::new(creator));
        }
        v
    }

    pub fn update(&self, grad: &Tensor, lr: f32) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(&grad, lr);
    }
}

impl Clone for Optimizee {
    fn clone(&self) -> Optimizee {
        Optimizee {
            inner: self.inner.clone(),
        }
    }
}

impl Backward for Optimizee {
    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]
        vec![]
    }

    fn get_optimizee(&self) -> Option<Optimizee> {
        Some(Optimizee {
            inner: self.inner.clone(),
        })
    }
}

pub fn optimize(loss: &Tensor, lr: f32) {
    let funcalles = graph::collect_funcalls(vec![loss.clone()]);
    let mut optimizees = Vec::new();
    let mut trainables = Vec::new();
    for fc in funcalles {
        if let Some(o) = fc.backward.get_optimizee() {
            optimizees.push(o);
            trainables.push(fc.get_ys()[0].clone());
        }
    }
    let grads = gradients(&vec![loss.clone()], &trainables, false);
    for (optimizee, grad) in optimizees.iter().zip(grads.iter()) {
        optimizee.update(grad, lr);
    }
}
