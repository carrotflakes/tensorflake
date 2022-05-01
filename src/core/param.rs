use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use super::{Backward, Computed, FunctionCall, NDArray, Optimizer};

pub trait ParamInnerT: Sync + Send + 'static {
    fn get(&self) -> Computed;
    fn set(&mut self, data: Computed);
    fn update(&mut self, grad: &NDArray);
    fn name(&self) -> Cow<'static, str>;

    fn create_graph(&self) -> bool {
        true
    }
}

struct ParamInner<T: Optimizer + Clone> {
    data: Computed,
    name: Cow<'static, str>,
    optimizer: T,
    state: T::State,
}

impl<T: Optimizer + Clone> ParamInnerT for ParamInner<T> {
    fn get(&self) -> Computed {
        self.data.clone()
    }

    fn set(&mut self, data: Computed) {
        self.data = data;
    }

    fn update(&mut self, grad: &NDArray) {
        self.optimizer
            .update(&mut self.data, &mut self.state, grad);
    }

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }
}

struct ParamInnerShared<T: Optimizer + Clone> {
    data: Computed,
    name: Cow<'static, str>,
    optimizer: Arc<Mutex<T>>,
    state: T::State,
}

impl<T: Optimizer + Clone> ParamInnerT for ParamInnerShared<T> {
    fn get(&self) -> Computed {
        self.data.clone()
    }

    fn set(&mut self, data: Computed) {
        self.data = data;
    }

    fn update(&mut self, grad: &NDArray) {
        let mut optimizer = self.optimizer.lock().unwrap().clone();
        optimizer.update(&mut self.data, &mut self.state, grad);
    }

    fn name(&self) -> Cow<'static, str> {
        self.name.clone()
    }
}

struct ParamInnerFixed {
    data: Computed,
    name: Cow<'static, str>,
}

impl ParamInnerT for ParamInnerFixed {
    fn get(&self) -> Computed {
        self.data.clone()
    }

    fn set(&mut self, data: Computed) {
        self.data = data;
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
                data: ndarray.into(),
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
                data: ndarray.into(),
            })),
        }
    }

    pub fn new_fixed(ndarray: NDArray, name: Cow<'static, str>) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInnerFixed {
                data: ndarray.into(),
                name,
            })),
        }
    }

    pub fn from_inner(inner: impl ParamInnerT) -> Param {
        Param {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn get(&self) -> Computed {
        let inner = self.inner.lock().unwrap();
        let v = inner.get().clone();
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
        inner.set(Computed::new(ndarray));
    }

    pub fn update(&self, grad: &NDArray) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(grad);
    }

    pub fn name(&self) -> Cow<'static, str> {
        let inner = self.inner.lock().unwrap();
        inner.name()
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
    fn backward(
        &self,
        xs: &Vec<Computed>,
        ys: &Vec<Computed>,
        gys: &Vec<Computed>,
    ) -> Vec<Computed> {
        #![allow(unused_variables)]
        vec![]
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn get_function_name(&self) -> Cow<'static, str> {
        self.inner.lock().unwrap().name()
    }
}
