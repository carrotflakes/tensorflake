use ndarray::Array;
use ndarray_rand::{rand::Rng, rand_distr::Uniform, RandomExt};

use super::Layer;
use crate::{functions::*, *};

pub struct Linear {
    pub w: Variable<ENABLE_BACKPROP>,
    pub b: Variable<ENABLE_BACKPROP>,
}

impl Linear {
    pub fn new(input: usize, output: usize, rng: &mut impl Rng) -> Self {
        Self {
            w: Variable::new(
                Array::random_using((input, output), Uniform::new(0., 0.01), rng).into_dyn(),
            )
            .named("w"),
            b: Variable::new(Array::zeros(output).into_dyn()).named("b"),
        }
    }
}

impl Layer for Linear {
    fn call<const ENABLE_BACKPROP: bool>(
        &self,
        xs: Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>
    where
        Self: Sized + 'static,
    {
        vec![call!(
            Add,
            call!(Matmul, xs[0], self.w.flip_bp()),
            self.b.flip_bp()
        )]
    }

    fn all_params(&self) -> Vec<Variable<ENABLE_BACKPROP>> {
        vec![self.w.clone(), self.b.clone()]
    }
}
