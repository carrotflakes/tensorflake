use crate::*;

pub fn optimize(loss: &Tensor, lr: f32) {
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
    for (param, grad) in params.iter().zip(grads.iter()) {
        param.update(grad, lr);
    }
}

#[derive(Default)]
pub struct GradientsAccumulator {
    pub table: std::collections::HashMap<Param, Tensor>,
}

impl GradientsAccumulator {
    pub fn compute(&mut self, loss: &Tensor) {
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

pub struct ExecutionContext {
    pub total: Option<usize>,
    pub epoch: usize,
    pub train: bool,
    pub loss: f32,
    pub processed: usize,
    pub corrected: usize,
}

impl ExecutionContext {
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
            "{} epoch: {}, loss: {:.4}, acc: {:.4}",
            if self.train { "train" } else { "valid" },
            self.epoch,
            self.loss / self.processed as f32,
            self.corrected as f32 / self.processed as f32
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
            total: self.data_len,
            epoch: self.current_epoch,
            train: self.train,
            loss: 0.0,
            processed: 0,
            corrected: 0,
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
