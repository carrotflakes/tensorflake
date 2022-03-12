use crate::{Backward, Tensor, Variable};

pub struct Funcall {
    pub(crate) function: Box<dyn Backward>,
    pub(crate) input: Vec<Variable<true>>,
    pub(crate) output: Vec<Variable<true>>,
    pub(crate) generation: u32,
}

impl Funcall {
    pub fn new(
        function: Box<dyn Backward>,
        input: Vec<Variable<true>>,
        output: Vec<Tensor>,
    ) -> Self {
        let gen = Variable::get_next_gen(&input);
        Self {
            function,
            input,
            output: output
                .into_iter()
                .map(|x| Variable::new_with_gen(x, gen))
                .collect(),
            generation: gen,
        }
    }

    pub(crate) fn backward(&self, retain_grad: bool, enable_backprop: bool) {
        let gys = self
            .output
            .iter()
            .map(|y| y.get_grad().expect("ensure terminal variable's grad"))
            .collect();

        let gxs = self.function.backward(&self.input, &gys, enable_backprop);

        for (x, gx) in self.input.iter().zip(gxs) {
            x.add_grad(gx);
        }

        if !retain_grad {
            for y in &self.output {
                y.clear_grad();
            }
        }
    }
}
