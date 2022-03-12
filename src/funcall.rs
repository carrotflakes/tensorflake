use crate::{Function, Variable};

pub struct Funcall {
    pub(crate) function: Box<dyn Function>,
    pub(crate) input: Vec<Variable>,
    pub(crate) output: Vec<Variable>,
}

impl Funcall {
    pub fn new(function: Box<dyn Function>, input: Vec<Variable>) -> Self {
        let output = function.forward(&input);
        let gen = Variable::get_next_gen(&input);
        Self {
            function,
            input,
            output: output
                .into_iter()
                .map(|x| Variable::new_with_gen(x, gen))
                .collect(),
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
