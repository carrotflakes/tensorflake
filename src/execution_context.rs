use crate::metrics::{Metric, Metrics};

pub struct ExecutionContext {
    pub total: Option<usize>,
    pub epoch: usize,
    pub train: bool,
    pub metrics: Metrics,
    pub time: std::time::Instant,
}

impl ExecutionContext {
    pub fn count(&mut self, n: usize) {
        self.metrics.count(n);
    }

    pub fn add_metric<T: Metric>(&mut self, metric: T) {
        self.metrics.add(metric);
    }

    pub fn merge_metrics(&mut self, metrics: Metrics) {
        self.metrics.merge(metrics);
    }

    pub fn print_progress(&self) {
        print!("\x1b[2K");
        if let Some(total) = self.total {
            print!("{:>6.2}%", self.metrics.total as f32 * 100.0 / total as f32);
        } else {
            print!("{}", self.metrics.total);
        }
        print!("\r");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    pub fn print_result(&self) {
        println!(
            "{} epoch: {}, {}time: {:.2}s",
            if self.train { "train" } else { "valid" },
            self.epoch,
            self.metrics.display_metrics(),
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
            metrics: Metrics::new(),
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
