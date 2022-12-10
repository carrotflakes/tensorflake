use std::sync::{Arc, Mutex};

use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Distribution,
    RandomExt,
};

use crate::{DefaultRng, NDArray, Optimizer, ParamNDA};

pub trait Initializer {
    fn initialize(&self, shape: &[usize]) -> ParamNDA;
    fn scope(&self, name: impl ToString) -> Self;
}

pub struct InitializerWithOptimizer<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> {
    pub distribution: D,
    pub optimizer: O,
    pub rng: Arc<Mutex<DefaultRng>>,
    pub path: Vec<String>,
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> InitializerWithOptimizer<D, O> {
    pub fn new(distribution: D, optimizer: O) -> Self {
        Self {
            distribution,
            optimizer,
            rng: Arc::new(Mutex::new(DefaultRng::seed_from_u64(42))),
            path: Default::default(),
        }
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> Initializer
    for InitializerWithOptimizer<D, O>
{
    fn initialize(&self, shape: &[usize]) -> ParamNDA {
        let mut rng = self.rng.lock().unwrap();
        let t = NDArray::random_using(shape, self.distribution.clone(), &mut *rng);
        ParamNDA::new(t, self.path.join(":").into(), self.optimizer.clone())
    }

    fn scope(&self, name: impl ToString) -> Self {
        let mut i = self.clone();
        i.path.push(name.to_string());
        i
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> Clone
    for InitializerWithOptimizer<D, O>
{
    fn clone(&self) -> Self {
        let rng = self.rng.clone();
        rng.lock().unwrap().gen::<u32>();
        Self {
            distribution: self.distribution.clone(),
            optimizer: self.optimizer.clone(),
            rng,
            path: self.path.clone(),
        }
    }
}

pub struct InitializerWithSharedOptimizer<
    D: Distribution<f32> + Clone,
    O: Optimizer<NDArray> + Clone,
> {
    pub distribution: D,
    pub optimizer: Arc<Mutex<O>>,
    pub rng: Arc<Mutex<DefaultRng>>,
    pub path: Vec<String>,
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone>
    InitializerWithSharedOptimizer<D, O>
{
    pub fn new(distribution: D, optimizer: Arc<Mutex<O>>) -> Self {
        Self {
            distribution,
            optimizer,
            rng: Arc::new(Mutex::new(DefaultRng::seed_from_u64(42))),
            path: Default::default(),
        }
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> Initializer
    for InitializerWithSharedOptimizer<D, O>
{
    fn initialize(&self, shape: &[usize]) -> ParamNDA {
        let mut rng = self.rng.lock().unwrap();
        let t = NDArray::random_using(shape, self.distribution.clone(), &mut *rng);
        ParamNDA::new_shared(t, self.path.join(":").into(), self.optimizer.clone())
    }

    fn scope(&self, name: impl ToString) -> Self {
        let mut i = self.clone();
        i.path.push(name.to_string());
        i
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer<NDArray> + Clone> Clone
    for InitializerWithSharedOptimizer<D, O>
{
    fn clone(&self) -> Self {
        let rng = self.rng.clone();
        rng.lock().unwrap().gen::<u32>();
        Self {
            distribution: self.distribution.clone(),
            optimizer: self.optimizer.clone(),
            rng,
            path: self.path.clone(),
        }
    }
}
