use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use super::{Backward, Computed, FunctionCall, NDArray, Optimizer};

pub trait OptimizerStateT: Sync + Send + 'static {
    fn update(&mut self, data: &mut NDArray, grad: &NDArray);
}

pub struct OptimizerState<T: Optimizer + Clone> {
    optimizer: T,
    state: T::State,
}

impl<T: Optimizer + Clone> OptimizerStateT for OptimizerState<T> {
    fn update(&mut self, data: &mut NDArray, grad: &NDArray) {
        self.optimizer.update(data, &mut self.state, grad);
    }
}

pub struct SharedOptimizerState<T: Optimizer + Clone> {
    optimizer: Arc<Mutex<T>>,
    state: T::State,
}

impl<T: Optimizer + Clone> OptimizerStateT for SharedOptimizerState<T> {
    fn update(&mut self, data: &mut NDArray, grad: &NDArray) {
        let mut optimizer = self.optimizer.lock().unwrap().clone();
        optimizer.update(data, &mut self.state, grad);
    }
}

struct ParamInner {
    data: NDArray,
    computed: Option<Computed>,
    name: Cow<'static, str>,
    optimizer_state: Box<dyn OptimizerStateT>,
}

impl ParamInner {
    fn new(
        data: NDArray,
        name: Cow<'static, str>,
        optimizer_state: Box<dyn OptimizerStateT>,
    ) -> Self {
        Self {
            data,
            computed: None,
            name,
            optimizer_state,
        }
    }

    fn update(&mut self, grad: &NDArray) {
        self.optimizer_state.update(&mut self.data, grad);
        self.computed = None;
    }
}

pub struct Param {
    inner: Arc<Mutex<ParamInner>>,
}

impl Param {
    pub fn new(
        ndarray: NDArray,
        name: Cow<'static, str>,
        optimizer: impl Optimizer + Clone,
    ) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(
                ndarray.clone().into(),
                name,
                Box::new(OptimizerState {
                    state: optimizer.new_state(&ndarray.shape()),
                    optimizer,
                }),
            ))),
        }
    }

    pub fn new_shared<T: Optimizer + Clone>(
        ndarray: NDArray,
        name: Cow<'static, str>,
        optimizer: Arc<Mutex<T>>,
    ) -> Param {
        let state = optimizer.lock().unwrap().new_state(&ndarray.shape());
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(
                ndarray.clone().into(),
                name,
                Box::new(SharedOptimizerState { optimizer, state }),
            ))),
        }
    }

    pub fn from_inner(
        ndarray: NDArray,
        name: Cow<'static, str>,
        optimizer_state: impl OptimizerStateT,
    ) -> Param {
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(
                ndarray.into(),
                name,
                Box::new(optimizer_state),
            ))),
        }
    }

    pub fn get(&self) -> Computed {
        let mut inner = self.inner.lock().unwrap();
        if inner.computed.is_none() {
            let computed = Computed::new(inner.data.clone());
            let creator = FunctionCall {
                backward: Box::new(Param {
                    inner: self.inner.clone(),
                }),
                xs: vec![],
                ys: vec![std::sync::Arc::downgrade(&computed.inner)],
            };
            computed.inner.attrs.lock().unwrap().creator = Some(std::sync::Arc::new(creator));
            inner.computed = Some(computed);
        }
        inner.computed.clone().unwrap()
    }

    pub fn set(&mut self, ndarray: NDArray) {
        let mut inner = self.inner.lock().unwrap();
        inner.data = ndarray;
    }

    pub fn update(&self, grad: &NDArray) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(grad);
    }

    pub fn name(&self) -> Cow<'static, str> {
        let inner = self.inner.lock().unwrap();
        inner.name.clone()
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
        self.inner.lock().unwrap().name.clone()
    }
}
