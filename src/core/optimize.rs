use super::{gradients, graph, Computed, Param};

pub fn optimize<T: Clone + Default + Send + Sync + 'static>(loss: &Computed<T>)
where
    T: graph::One + for<'a> std::ops::Add<&'a T, Output = T>,
    for<'a> &'a Computed<T>: std::ops::Add<&'a Computed<T>, Output = Computed<T>>,
{
    let mut ga = GradientsAccumulator::new();
    ga.compute(loss);
    ga.optimize();
}

pub struct GradientsAccumulator<T: Default + Send + Sync + 'static> {
    pub table: std::collections::HashMap<Param<T>, T>,
}

impl<T: Clone + Default + Send + Sync + 'static> GradientsAccumulator<T>
where
    T: graph::One + for<'a> std::ops::Add<&'a T, Output = T>,
    for<'a> &'a Computed<T>: std::ops::Add<&'a Computed<T>, Output = Computed<T>>,
{
    pub fn new() -> Self {
        Self {
            table: std::collections::HashMap::new(),
        }
    }

    pub fn compute(&mut self, loss: &Computed<T>) {
        let (params, grads) = collect_params_grads(loss);
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            self.push(param, (*grad).clone());
        }
    }

    pub fn push(&mut self, param: Param<T>, grad: T) {
        match self.table.entry(param) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                *e.get_mut() = grad + e.get();
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

fn collect_params_grads<T: Clone + Default + Send + Sync + 'static>(
    loss: &Computed<T>,
) -> (Vec<Param<T>>, Vec<Computed<T>>)
where
    T: graph::One,
    for<'a> &'a Computed<T>: std::ops::Add<&'a Computed<T>, Output = Computed<T>>,
{
    let function_calls = graph::collect_function_calls(vec![loss.clone()]);

    let mut params = Vec::new();
    let mut trainables = Vec::new();

    for fc in function_calls {
        if let Some(o) = fc
            .backward
            .as_any()
            .and_then(|o| o.downcast_ref::<Param<T>>())
            .cloned()
        {
            let trainable = fc.get_ys().pop().unwrap();
            if trainables.contains(&trainable) {
                panic!("same trainables");
                // continue;
            }
            trainables.push(trainable);
            params.push(o);
        }
    }

    let grads = gradients(&[loss.clone()], &trainables, false);
    (params, grads)
}
