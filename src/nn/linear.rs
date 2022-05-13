use super::Layer;
use crate::{functions::*, initializers::Initializer, optimizers::Fixed, *};

#[derive(serde::Serialize, serde::Deserialize)]
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
                (*self.w.get()).clone(),
                self.w.get_function_name(),
                Fixed,
            ),
            b: self
                .b
                .as_ref()
                .map(|b| Param::new((*b.get()).clone(), self.w.get_function_name(), Fixed)),
        }
    }
}

impl Layer for Linear {
    type Input = Computed;
    type Output = Computed;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        if let Some(b) = &self.b {
            matmul_add(&x, &self.w.get(), &b.get())
        } else {
            x.matmul(&self.w.get())
        }
    }

    fn all_params(&self) -> Vec<Param> {
        [self.w.clone()].into_iter().chain(self.b.clone()).collect()
    }
}
