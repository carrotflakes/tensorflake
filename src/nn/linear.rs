use super::Layer;
use crate::{functions::*, initializers::Initializer, optimizers::Fixed, *};

pub struct Linear {
    pub w: Param,
    pub b: Param,
}

impl Linear {
    pub fn new(
        input: usize,
        output: usize,
        w: &mut impl Initializer,
        b: &mut impl Initializer,
    ) -> Self {
        Self {
            w: w.initialize(&[input, output]),
            b: b.initialize(&[output]),
        }
    }

    pub fn build(&self) -> Self {
        Self {
            w: Param::new((*self.w.get_tensor()).clone(), Fixed),
            b: Param::new((*self.b.get_tensor()).clone(), Fixed),
        }
    }
}

impl Layer for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        matmul_add(&x, &self.w.get_tensor(), &self.b.get_tensor())
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}
