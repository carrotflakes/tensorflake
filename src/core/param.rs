use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use super::{Backward, FunctionCall, NDArray, Optimizer, Tensor};

pub trait ParamInnerT: Sync + Send + 'static {
    fn tensor(&self) -> Tensor;
    fn set(&mut self, tensor: Tensor);
    fn update(&mut self, grad: &NDArray);
    fn name(&self) -> Cow<'static, str>;

    fn create_graph(&self) -> bool {
        true
    }
}

struct ParamInner<T: Optimizer + Clone> {
    tensor: Tensor,
    name: Cow<'static, str>,
    optimizer: T,
    state: T::State,
}

impl<T: Optimizer + Clone> ParamInnerT for ParamInner<T> {
    fn tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray) {
        self.optimizer
            .update(&mut self.tensor, &mut self.state, grad);
    }

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }
}

struct ParamInnerShared<T: Optimizer + Clone> {
    tensor: Tensor,
    name: Cow<'static, str>,
    optimizer: Arc<Mutex<T>>,
    state: T::State,
}

impl<T: Optimizer + Clone> ParamInnerT for ParamInnerShared<T> {
    fn tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray) {
        let mut optimizer = self.optimizer.lock().unwrap().clone();
        optimizer.update(&mut self.tensor, &mut self.state, grad);
    }

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }
}

struct ParamInnerFixed {
    tensor: Tensor,
    name: Cow<'static, str>,
}

impl ParamInnerT for ParamInnerFixed {
    fn tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    fn set(&mut self, tensor: Tensor) {
        self.tensor = tensor;
    }

    fn update(&mut self, grad: &NDArray) {
        drop(grad);
    }

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }

    fn create_graph(&self) -> bool {
        false
    }
}

pub struct Param {
    inner: Arc<Mutex<dyn ParamInnerT>>,
}

impl Param {
    pub fn new(
        ndarray: NDArray,
        name: Cow<'static, str>,
        optimizer: impl Optimizer + Clone,
    ) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInner {
                name,
                state: optimizer.new_state(&ndarray.shape()),
                optimizer,
                tensor: ndarray.into(),
            })),
        }
    }

    pub fn new_shared<T: Optimizer + Clone>(
        ndarray: NDArray,
        name: Cow<'static, str>,
        optimizer: Arc<Mutex<T>>,
    ) -> Param {
        let state = optimizer.lock().unwrap().new_state(&ndarray.shape());
        Param {
            inner: Arc::new(Mutex::new(ParamInnerShared {
                name,
                state,
                optimizer,
                tensor: ndarray.into(),
            })),
        }
    }

    pub fn new_fixed(ndarray: NDArray, name: Cow<'static, str>) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInnerFixed {
                tensor: ndarray.into(),
                name,
            })),
        }
    }

    pub fn from_inner(inner: impl ParamInnerT) -> Param {
        Param {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn get_tensor(&self) -> Tensor {
        let inner = self.inner.lock().unwrap();
        let v = inner.tensor().clone();
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

    pub fn update(&self, grad: &NDArray) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(grad);
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

    fn get_function_name(&self) -> Cow<'static, str> {
        self.inner.lock().unwrap().name()
    }
}
