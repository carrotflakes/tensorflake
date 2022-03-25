use super::Layer;
use crate::{functions::*, *};

pub struct Linear {
    pub w: Optimizee,
    pub b: Optimizee,
}

impl Linear {
    pub fn new(
        input: usize,
        output: usize,
        w: &mut impl FnMut(&[usize]) -> Optimizee,
        b: &mut impl FnMut(&[usize]) -> Optimizee,
    ) -> Self {
        Self {
            w: w(&[input, output]),
            b: b(&[output]),
        }
    }

    pub fn build(&self) -> Self {
        Self {
            w: Fixed::new((*self.w.get()).clone()),
            b: Fixed::new((*self.b.get()).clone()),
        }
    }
}

impl Layer for Linear {
    fn call(&self, xs: Vec<Tensor>, _train: bool) -> Vec<Tensor>
    where
        Self: Sized + 'static,
    {
        vec![call!(Add, call!(Matmul, xs[0], self.w.get()), self.b.get())]
    }

    fn all_optimizees(&self) -> Vec<Optimizee> {
        vec![self.w.clone(), self.b.clone()]
    }
}
