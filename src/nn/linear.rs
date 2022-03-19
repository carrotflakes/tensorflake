use ndarray::Array;
use ndarray_rand::{rand::Rng, rand_distr::Uniform, RandomExt};

use super::Layer;
use crate::{functions::*, *};

pub struct Linear {
    pub w: Box<dyn Fn() -> Variable>,
    pub b: Box<dyn Fn() -> Variable>,
}

impl Linear {
    pub fn new(
        input: usize,
        output: usize,
        param_gen: &(impl Fn(Tensor) -> Box<dyn Fn() -> Variable> + 'static),
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            w: param_gen(
                Array::random_using((input, output), Uniform::new(0., 0.01), rng).into_tensor(),
            ),
            b: param_gen(Array::zeros(output).into_tensor()),
        }
    }
}

impl Layer for Linear {
    fn call(&self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        vec![call!(Add, call!(Matmul, xs[0], (self.w)()), (self.b)())]
    }

    fn all_params(&self) -> Vec<Variable> {
        vec![(self.w)(), (self.b)()]
    }
}
