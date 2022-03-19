use ndarray::Array;
use ndarray_rand::{rand::Rng, rand_distr::Uniform, RandomExt};

use super::Layer;
use crate::{functions::*, *};

pub struct Linear {
    pub w: Variable,
    pub b: Variable,
}

impl Linear {
    pub fn new(input: usize, output: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: Variable::new(
                Array::random_using((input, output), Uniform::new(0., 0.01), rng).into_tensor(),
            )
            .named("w"),
            b: Variable::new(Array::zeros(output).into_tensor()).named("b"),
        }
    }
}

impl Layer for Linear {
    fn call(&self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        vec![call!(Add, call!(Matmul, xs[0], self.w), self.b)]
    }

    fn all_params(&self) -> Vec<Variable> {
        vec![self.w.clone(), self.b.clone()]
    }
}
