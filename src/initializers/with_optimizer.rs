use std::marker::PhantomData;

use super::{Initializer, Scope};
use crate::{graph::One, Optimizer, Param};

#[derive(Clone)]
pub struct InitializerWithOptimizer<
    T: 'static + One + Default + Clone + Send + Sync,
    I: Initializer<T> + Clone,
    O: Optimizer<T> + Clone,
> {
    pub initializer: I,
    pub optimizer: O,
    pub path: Vec<String>,
    pub(crate) _t: PhantomData<T>,
}

impl<
        T: 'static + One + Default + Clone + Send + Sync,
        I: Initializer<T> + Clone,
        O: Optimizer<T> + Clone,
    > InitializerWithOptimizer<T, I, O>
{
    pub fn new(initializer: I, optimizer: O) -> Self {
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
    > Initializer<Param<T>> for InitializerWithOptimizer<T, I, O>
{
    fn initialize(&self, shape: &[usize]) -> Param<T> {
        let data = self.initializer.initialize(shape);
        Param::new(data, self.path.join(":").into(), self.optimizer.clone())
    }
}

impl<
        T: 'static + One + Default + Clone + Send + Sync,
        I: Initializer<T> + Clone,
        O: Optimizer<T> + Clone,
    > Scope for InitializerWithOptimizer<T, I, O>
{
    fn scope(&self, name: impl ToString) -> Self {
        let mut i = self.clone();
        i.path.push(name.to_string());
        i
    }
}
