use std::sync::{Arc, Mutex};

use crate::*;

trait ParamInnerT: Sync + Send + 'static {
    fn tensor_ref(&self) -> &Tensor;
    fn set(&mut self, tensor: Tensor);
    fn update(&mut self, grad: &NDArray);

    fn create_graph(&self) -> bool {
        true
    }
}

struct ParamInner<T: Optimizer> {
    tensor: Tensor,
    optimizer: T,
    state: T::State,
}

impl<T: Optimizer> ParamInnerT for ParamInner<T> {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray) {
        self.optimizer
            .update(&mut self.tensor, &mut self.state, grad);
    }

    fn create_graph(&self) -> bool {
        self.optimizer.create_graph()
    }
}

struct ParamInnerShared<T: Optimizer> {
    tensor: Tensor,
    optimizer: Arc<Mutex<T>>,
    state: T::State,
}

impl<T: Optimizer> ParamInnerT for ParamInnerShared<T> {
    fn tensor_ref(&self) -> &Tensor {
        &self.tensor
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray) {
        let mut optimizer = self.optimizer.lock().unwrap().clone();
        optimizer.update(&mut self.tensor, &mut self.state, grad);
    }

    fn create_graph(&self) -> bool {
        self.optimizer.lock().unwrap().create_graph()
    }
}

pub struct Param {
    inner: Arc<Mutex<dyn ParamInnerT>>,
}

impl Param {
    pub fn new(ndarray: NDArray, optimizer: impl Optimizer) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInner {
                state: optimizer.new_state(&ndarray.shape()),
                optimizer,
                tensor: ndarray.into(),
            })),
        }
    }

    pub fn new_shared<T: Optimizer>(ndarray: NDArray, optimizer: Arc<Mutex<T>>) -> Param {
        let state = optimizer.lock().unwrap().new_state(&ndarray.shape());
        Param {
            inner: Arc::new(Mutex::new(ParamInnerShared {
                state,
                optimizer,
                tensor: ndarray.into(),
            })),
        }
    }

    pub fn get_tensor(&self) -> Tensor {
        let inner = self.inner.lock().unwrap();
        let v = inner.tensor_ref().clone();
        if inner.create_graph() && !v.has_creator() {
            let creator = FunctionCall {
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
        inner.set(Tensor::new(ndarray));
    }

    pub fn update(&self, grad: &Tensor) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(&grad);
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
