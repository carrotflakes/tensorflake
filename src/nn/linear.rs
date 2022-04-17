use super::Layer;
use crate::{functions::*, initializers::Initializer, optimizers::Fixed, *};

pub struct Linear {
    pub w: Param,
    pub b: Option<Param>,
}

impl Linear {
    pub fn new(
        input: usize,
        output: usize,
        w: impl Initializer,
        b: Option<impl Initializer>,
    ) -> Self {
        Self {
            w: w.initialize(&[input, output]),
            b: b.map(|b| b.initialize(&[output])),
        }
    }

    pub fn build(&self) -> Self {
        Self {
            w: Param::new(
                (*self.w.get_tensor()).clone(),
                self.w.get_function_name(),
                Fixed,
            ),
            b: self
                .b
                .as_ref()
                .map(|b| Param::new((*b.get_tensor()).clone(), self.w.get_function_name(), Fixed)),
        }
    }
}

impl Layer for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        if let Some(b) = &self.b {
            matmul_add(&x, &self.w.get_tensor(), &b.get_tensor())
        } else {
            x.matmul(&self.w.get_tensor())
        }
    }

    fn all_params(&self) -> Vec<Param> {
        [self.w.clone()].into_iter().chain(self.b.clone()).collect()
    }
}
