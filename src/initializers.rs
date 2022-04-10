use std::sync::{Arc, Mutex};

use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Distribution,
    RandomExt,
};

use crate::{DefaultRng, NDArray, Optimizer, Param};

pub trait Initializer {
    fn initialize(&mut self, shape: &[usize]) -> Param;
}

impl<F: FnMut(&[usize]) -> Param> Initializer for F {
    fn initialize(&mut self, shape: &[usize]) -> Param {
        (self)(shape)
    }
}

pub struct InitializerWithOptimizer<D: Distribution<f32> + Clone, O: Optimizer> {
    pub distribution: D,
    pub optimizer: O,
    pub rng: DefaultRng,
}

impl<D: Distribution<f32> + Clone, O: Optimizer> InitializerWithOptimizer<D, O> {
    pub fn new(distribution: D, optimizer: O) -> Self {
        Self {
            distribution,
            optimizer,
            rng: DefaultRng::seed_from_u64(42),
        }
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer> Initializer for InitializerWithOptimizer<D, O> {
    fn initialize(&mut self, shape: &[usize]) -> Param {
        let t = NDArray::random_using(shape, self.distribution.clone(), &mut self.rng);
        Param::new(t, self.optimizer.clone())
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer> Clone for InitializerWithOptimizer<D, O> {
    fn clone(&self) -> Self {
        let mut rng = self.rng.clone();
        rng.gen::<u32>();
        Self {
            distribution: self.distribution.clone(),
            optimizer: self.optimizer.clone(),
            rng,
        }
    }
}

pub struct InitializerWithSharedOptimizer<D: Distribution<f32> + Clone, O: Optimizer> {
    pub distribution: D,
    pub optimizer: Arc<Mutex<O>>,
    pub rng: DefaultRng,
}

impl<D: Distribution<f32> + Clone, O: Optimizer> InitializerWithSharedOptimizer<D, O> {
    pub fn new(distribution: D, optimizer: Arc<Mutex<O>>) -> Self {
        Self {
            distribution,
            optimizer,
            rng: DefaultRng::seed_from_u64(42),
        }
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer> Initializer
    for InitializerWithSharedOptimizer<D, O>
{
    fn initialize(&mut self, shape: &[usize]) -> Param {
        let t = NDArray::random_using(shape, self.distribution.clone(), &mut self.rng);
        Param::new_shared(t, self.optimizer.clone())
    }
}

impl<D: Distribution<f32> + Clone, O: Optimizer> Clone for InitializerWithSharedOptimizer<D, O> {
    fn clone(&self) -> Self {
        let mut rng = self.rng.clone();
        rng.gen::<u32>();
        Self {
            distribution: self.distribution.clone(),
            optimizer: self.optimizer.clone(),
            rng,
        }
    }
}
