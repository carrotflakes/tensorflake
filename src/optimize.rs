use crate::*;

pub fn optimize(loss: &Tensor) {
    let mut ga = GradientsAccumulator::new();
    ga.compute(loss);
    ga.optimize();
}

pub struct GradientsAccumulator {
    pub table: std::collections::HashMap<Param, NDArray>,
}

impl GradientsAccumulator {
    pub fn new() -> Self {
        Self {
            table: std::collections::HashMap::new(),
        }
    }

    pub fn compute(&mut self, loss: &Tensor) {
        let (params, grads) = collect_params_grads(loss);
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            self.push(param, (*grad).clone());
        }
    }

    pub fn push(&mut self, param: Param, grad: NDArray) {
        match self.table.entry(param) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                *e.get_mut() = e.get() + grad;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(grad);
            }
        }
    }

    pub fn optimize(&mut self) {
        for (param, grad) in &self.table {
            param.update(grad);
        }
        self.table.clear();
    }

    pub fn merge(&mut self, mut other: Self) {
        for (param, grad) in other.table.drain() {
            self.push(param, grad);
        }
    }
}

fn collect_params_grads(loss: &Tensor) -> (Vec<Param>, Vec<Tensor>) {
    let function_calls = graph::collect_function_calls(vec![loss.clone()]);
    let mut params = Vec::new();
    let mut trainables = Vec::new();
    for fc in function_calls {
        if let Some(o) = fc.backward.get_param() {
            params.push(o);
            trainables.push(fc.get_ys().pop().unwrap());
        }
    }
    let grads = gradients(&[loss.clone()], &trainables, false);
    (params, grads)
}
