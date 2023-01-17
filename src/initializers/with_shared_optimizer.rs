use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use super::{Initializer, Scope};
use crate::{graph::One, Optimizer, Param};

#[derive(Clone)]
pub struct InitializerWithSharedOptimizer<
    T: 'static + One + Default + Clone + Send + Sync,
    I: Initializer<T> + Clone,
    O: Optimizer<T> + Clone,
> {
    pub initializer: I,
    pub optimizer: Arc<Mutex<O>>,
    pub path: Vec<String>,
    pub(crate) _t: PhantomData<T>,
}

impl<
        T: 'static + One + Default + Clone + Send + Sync,
        I: Initializer<T> + Clone,
        O: Optimizer<T> + Clone,
    > InitializerWithSharedOptimizer<T, I, O>
{
    pub fn new(initializer: I, optimizer: Arc<Mutex<O>>) -> Self {
        Self {
            initializer,
            optimizer,
            path: Default::default(),
            _t: Default::default(),
        }
    }
}

impl<
        T: 'static + One + Default + Clone + Send + Sync,
        I: Initializer<T> + Clone,
        O: Optimizer<T> + Clone,
    > Initializer<Param<T>> for InitializerWithSharedOptimizer<T, I, O>
{
    fn initialize(&self, shape: &[usize]) -> Param<T> {
        let data = self.initializer.initialize(shape);
        Param::new_shared(data, self.path.join(":").into(), self.optimizer.clone())
    }
}

impl<
        T: 'static + One + Default + Clone + Send + Sync,
        I: Initializer<T> + Clone,
        O: Optimizer<T> + Clone,
    > Scope for InitializerWithSharedOptimizer<T, I, O>
{
    fn scope(&self, name: impl ToString) -> Self {
        let mut i = self.clone();
        i.path.push(name.to_string());
        i
    }
}
