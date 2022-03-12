use crate::{Function, Variable};

pub struct Funcall {
    pub(crate) function: Box<dyn Function>,
    pub(crate) input: Vec<Variable>,
    pub(crate) output: Vec<Variable>,
}

impl Funcall {
    pub fn new(function: Box<dyn Function>, input: Vec<Variable>) -> Self {
        let output = function.forward(&input);
        Self {
            function,
            input,
            output,
        }
    }

    pub fn backward(&self) {
        let gys = self.output.iter().map(|y| y.get_grad().unwrap()).collect();
        let gxs = self.function.backward(&self.input, &gys);
        for (x, gx) in self.input.iter().zip(gxs) {
            x.add_grad(gx);
        }
    }
}
