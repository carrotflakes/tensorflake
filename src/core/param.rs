use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

use super::{graph::One, Backward, Computed, FunctionCall, Optimizer};

pub trait OptimizerStateT<T: Sync + Send + 'static>: Sync + Send + 'static {
    fn update(&mut self, data: &mut T, grad: &T);
}

pub struct OptimizerState<T: Sync + Send + 'static, O: Optimizer<T> + Clone> {
    optimizer: O,
    state: O::State,
}

impl<T: Sync + Send + 'static, O: Optimizer<T> + Clone> OptimizerStateT<T>
    for OptimizerState<T, O>
{
    fn update(&mut self, data: &mut T, grad: &T) {
        self.optimizer.update(data, &mut self.state, grad);
    }
}

pub struct SharedOptimizerState<T: Sync + Send + 'static, O: Optimizer<T> + Clone> {
    optimizer: Arc<Mutex<O>>,
    state: O::State,
}

impl<T: Sync + Send + 'static, O: Optimizer<T> + Clone> OptimizerStateT<T>
    for SharedOptimizerState<T, O>
{
    fn update(&mut self, data: &mut T, grad: &T) {
        let mut optimizer = self.optimizer.lock().unwrap().clone();
        optimizer.update(data, &mut self.state, grad);
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ParamInner<T: Send + Sync + 'static> {
    data: T,
    #[serde(skip)]
    computed: Option<Computed<T>>,
    name: Cow<'static, str>,
    #[serde(skip, default = "default_optimizer_state")]
    optimizer_state: Box<dyn OptimizerStateT<T>>,
}

fn default_optimizer_state<T: Send + Sync + 'static>() -> Box<dyn OptimizerStateT<T>> {
    Box::new(OptimizerState {
        optimizer: crate::optimizers::Fixed,
        state: Default::default(),
    })
}

impl<T: Send + Sync + 'static> ParamInner<T> {
    fn new(data: T, name: Cow<'static, str>, optimizer_state: Box<dyn OptimizerStateT<T>>) -> Self {
        Self {
            data,
            computed: None,
            name,
            optimizer_state,
        }
    }

    fn update(&mut self, grad: &T) {
        self.optimizer_state.update(&mut self.data, grad);
        self.computed = None;
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Param<T: Default + Send + Sync + 'static> {
    inner: Arc<Mutex<ParamInner<T>>>,
}

impl<T: Clone + Default + Send + Sync + 'static + One> Param<T> {
    pub fn new(data: T, name: Cow<'static, str>, optimizer: impl Optimizer<T> + Clone) -> Self {
        let optimizer_state = Box::new(OptimizerState {
            state: optimizer.new_state(&data.shape()),
            optimizer,
        });
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(data, name, optimizer_state))),
        }
    }

    pub fn new_shared<O: Optimizer<T> + Clone>(
        data: T,
        name: Cow<'static, str>,
        optimizer: Arc<Mutex<O>>,
    ) -> Self {
        let state = optimizer.lock().unwrap().new_state(&data.shape());
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(
                data,
                name,
                Box::new(SharedOptimizerState { optimizer, state }),
            ))),
        }
    }

    pub fn from_inner(
        data: T,
        name: Cow<'static, str>,
        optimizer_state: impl OptimizerStateT<T>,
    ) -> Self {
        Param {
            inner: Arc::new(Mutex::new(ParamInner::new(
                data,
                name,
                Box::new(optimizer_state),
            ))),
        }
    }

    pub fn get(&self) -> Computed<T> {
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

    pub fn set(&mut self, data: T) {
        let mut inner = self.inner.lock().unwrap();
        inner.data = data;
    }

    pub fn update(&self, grad: &T) {
        let mut inner = self.inner.lock().unwrap();
        inner.update(grad);
    }

    pub fn name(&self) -> Cow<'static, str> {
        let inner = self.inner.lock().unwrap();
        inner.name.clone()
    }
}

impl<T: Default + Send + Sync + 'static> Clone for Param<T> {
    fn clone(&self) -> Param<T> {
        Param {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Default + Send + Sync + 'static> PartialEq for Param<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: Default + Send + Sync + 'static> Eq for Param<T> {}

impl<T: Default + Send + Sync + 'static> std::hash::Hash for Param<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl<T: Default + Send + Sync + 'static> Backward<T> for Param<T> {
    fn backward(
        &self,
        xs: &Vec<Computed<T>>,
        ys: &Vec<Computed<T>>,
        gys: &Vec<Computed<T>>,
    ) -> Vec<Computed<T>> {
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
