use std::sync::{Arc, Mutex};

use crate::*;

pub trait OptimizeeT: 'static {
    fn tensor_ref(&self) -> &Tensor;
    fn update(&mut self, grad: &Tensor, lr: f32);
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

    pub fn get(&self) -> Variable {
        let v = Variable::new(self.inner.lock().unwrap().tensor_ref().clone());
        let creator = Funcall {
            backward: Box::new(OptimizeeCreator {
                optimizee: Optimizee {
                    inner: self.inner.clone(),
                },
            }),
            xs: vec![],
            ys: vec![std::sync::Arc::downgrade(&v.inner)],
        };
        v.inner.attrs.lock().unwrap().creator = Some(std::sync::Arc::new(creator));
        v
    }

    pub fn update(&self, grad: &Variable, lr: f32) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(&grad, lr);
    }
}

struct OptimizeeCreator {
    optimizee: Optimizee,
}

impl Backward for OptimizeeCreator {
    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]
        vec![]
    }

    fn get_optimizee(&self) -> Option<Optimizee> {
        Some(Optimizee {
            inner: self.optimizee.inner.clone(),
        })
    }
}

pub fn optimize(loss: &Variable, lr: f32) {
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
