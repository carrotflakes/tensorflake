use super::Layer;
use crate::{functions::*, *};

pub struct Linear {
    pub w: Param,
    pub b: Param,
}

impl Linear {
    pub fn new(
        input: usize,
        output: usize,
        w: &mut impl FnMut(&[usize]) -> Param,
        b: &mut impl FnMut(&[usize]) -> Param,
    ) -> Self {
        Self {
            w: w(&[input, output]),
            b: b(&[output]),
        }
    }

    pub fn build(&self) -> Self {
        Self {
            w: Fixed::new((*self.w.get_tensor()).clone()),
            b: Fixed::new((*self.b.get_tensor()).clone()),
        }
    }
}

impl Layer for Linear {
    fn call(&self, xs: Vec<Tensor>, _train: bool) -> Vec<Tensor>
    where
        Self: Sized + 'static,
    {
        vec![call!(Add, call!(Matmul, xs[0], self.w.get_tensor()), self.b.get_tensor())]
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}
