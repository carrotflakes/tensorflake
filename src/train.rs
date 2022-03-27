use crate::{metrics::Metric, *};

pub fn optimize(loss: &Tensor, lr: f32) {
    let mut ga = GradientsAccumulator::new();
    ga.compute(loss);
    ga.optimize(lr);
}

pub struct GradientsAccumulator {
    pub table: std::collections::HashMap<Param, Tensor>,
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
            self.push(param, grad);
        }
    }

    pub fn push(&mut self, param: Param, grad: Tensor) {
        match self.table.entry(param) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                *e.get_mut() = e.get() + &grad;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(grad.clone());
            }
        }
    }

    pub fn optimize(&mut self, lr: f32) {
        for (param, grad) in &self.table {
            param.update(grad, lr);
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
    let funcalles = graph::collect_funcalls(vec![loss.clone()]);
    let mut params = Vec::new();
    let mut trainables = Vec::new();
    for fc in funcalles {
        if let Some(o) = fc.backward.get_param() {
            params.push(o);
            trainables.push(fc.get_ys()[0].clone());
        }
    }
    let grads = gradients(&vec![loss.clone()], &trainables, false);
    (params, grads)
}

pub struct ExecutionContext {
    pub total: Option<usize>,
    pub epoch: usize,
    pub train: bool,
    pub processed: usize,
    pub metrics: Vec<(std::any::TypeId, Box<dyn Metric>)>,
    time: std::time::Instant,
}

impl ExecutionContext {
    pub fn count(&mut self, n: usize) {
        self.processed += n;
    }

    pub fn push_metric<T: Metric + 'static>(&mut self, metric: T) {
        let id = std::any::TypeId::of::<T>();
        for (t, m) in &mut self.metrics {
            if id == *t {
                let m = (m as &mut dyn std::any::Any).downcast_mut::<T>().unwrap();
                m.merge(&metric);
                return;
            }
        }
        self.metrics.push((id, Box::new(metric)));
    }

    pub fn display_metrics(&self) -> String {
        let mut s = String::new();
        for (_, m) in &self.metrics {
            s += &format!("{}: {:.4}, ", m.name(), m.value() / self.processed as f32);
        }
        s
    }

    pub fn print_progress(&self) {
        if let Some(total) = self.total {
            print!("{:>6.2}%", self.processed as f32 * 100.0 / total as f32);
        } else {
            print!("{}", self.processed);
        }
        print!("\r");
    }

    pub fn print_result(&self) {
        println!(
            "{} epoch: {}, {}time: {:.2}s",
            if self.train { "train" } else { "valid" },
            self.epoch,
            self.display_metrics(),
            self.time.elapsed().as_secs_f32()
        );
    }
}

pub struct ExecutionContextIter {
    pub total_epochs: usize,
    pub data_len: Option<usize>,
    pub current_epoch: usize,
    pub train: bool,
}

impl ExecutionContextIter {
    pub fn new(total_epochs: usize, data_len: Option<usize>) -> Self {
        Self {
            total_epochs,
            data_len,
            current_epoch: 1,
            train: true,
        }
    }
}

impl Iterator for ExecutionContextIter {
    type Item = ExecutionContext;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_epoch > self.total_epochs {
            return None;
        }

        let ctx = ExecutionContext {
            total: if self.train { self.data_len } else { None },
            epoch: self.current_epoch,
            train: self.train,
            processed: 0,
            metrics: Vec::new(),
            time: std::time::Instant::now(),
        };

        if self.train {
            self.train = false;
        } else {
            self.current_epoch += 1;
            self.train = true;
        }

        Some(ctx)
    }
}

#[test]
fn test_execution_context_iter() {
    for c in ExecutionContextIter::new(2, None) {
        c.print_result();
    }
}
