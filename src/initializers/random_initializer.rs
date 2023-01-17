use super::Initializer;
use crate::{DefaultRng, NDArray};

use std::sync::Arc;
use std::sync::Mutex;

use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Distribution,
    RandomExt,
};

pub struct RandomInitializer<D: Distribution<f32> + Clone> {
    pub distribution: D,
    pub rng: Arc<Mutex<DefaultRng>>,
}

impl<D: Distribution<f32> + Clone> RandomInitializer<D> {
    pub fn new(distribution: D) -> Self {
        Self {
            distribution,
            rng: Arc::new(Mutex::new(DefaultRng::seed_from_u64(42))),
        }
    }
}

impl<D: Distribution<f32> + Clone> Initializer<NDArray> for RandomInitializer<D> {
    fn initialize(&self, shape: &[usize]) -> NDArray {
        let mut rng = self.rng.lock().unwrap();
        NDArray::random_using(shape, self.distribution.clone(), &mut *rng)
    }
}

impl<D: Distribution<f32> + Clone> Clone for RandomInitializer<D> {
    fn clone(&self) -> Self {
        let rng = self.rng.clone();
        rng.lock().unwrap().gen::<u32>();
        Self {
            distribution: self.distribution.clone(),
            rng,
        }
    }
}
