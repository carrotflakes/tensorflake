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
        w: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
        b: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
    ) -> Self {
        Self {
            w: w(&[input, output]),
            b: b(&[output]),
        }
    }

    pub fn build(&self) -> Self {
        let w = (self.w)();
        let b = (self.b)();
        Self {
            w: Box::new(move || w.clone()),
            b: Box::new(move || b.clone()),
        }
    }
}

impl Layer for Linear {
    fn call(&self, xs: Vec<Variable>, _train: bool) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        vec![call!(Add, call!(Matmul, xs[0], (self.w)()), (self.b)())]
    }

    fn all_params(&self) -> Vec<Variable> {
        vec![(self.w)(), (self.b)()]
    }
}
