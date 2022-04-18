use ndarray::s;

use crate::*;

pub fn argmax_accuracy(t: &[usize], y: &Computed) -> Accuracy {
    let y = y
        .view()
        .into_shape([
            y.shape().iter().take(y.ndim() - 1).product::<usize>(),
            y.shape()[y.ndim() - 1],
        ])
        .unwrap();

    let correct = t
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            let y = y
                .slice(s![*i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            y == **t
        })
        .count();

    Accuracy::new(correct)
}

pub trait Metric: Sync + Send + 'static {
    fn name(&self) -> &'static str;
    fn value(&self) -> f32;
    fn merge(&mut self, other: &Self);
    fn as_any(&self) -> &dyn std::any::Any;
}

trait MetricObjectSafe: Sync + Send {
    fn name(&self) -> &'static str;
    fn value(&self) -> f32;
    fn merge(&mut self, other: &dyn MetricObjectSafe);
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: Metric> MetricObjectSafe for T {
    fn name(&self) -> &'static str {
        T::name(self)
    }

    fn value(&self) -> f32 {
        T::value(self)
    }

    fn merge(&mut self, other: &dyn MetricObjectSafe) {
        T::merge(self, other.as_any().downcast_ref::<T>().unwrap())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self.as_any()
    }
}

#[derive(Clone, Default)]
pub struct Loss {
    acc_loss: f32,
}

impl Loss {
    pub fn new(loss: f32, count: usize) -> Loss {
        Loss { acc_loss: loss * count as f32 }
    }
}

impl Metric for Loss {
    fn name(&self) -> &'static str {
        "loss"
    }

    fn value(&self) -> f32 {
        self.acc_loss as f32
    }

    fn merge(&mut self, other: &Self) {
        self.acc_loss += other.acc_loss;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone, Default)]
pub struct Accuracy {
    correct: usize,
}

impl Accuracy {
    pub fn new(correct: usize) -> Accuracy {
        Accuracy { correct }
    }
}

impl Metric for Accuracy {
    fn name(&self) -> &'static str {
        "accuracy"
    }

    fn value(&self) -> f32 {
        self.correct as f32
    }

    fn merge(&mut self, other: &Self) {
        self.correct += other.correct;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct Metrics {
    pub total: usize,
    metrics: Vec<(std::any::TypeId, Box<dyn MetricObjectSafe>)>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            total: 0,
            metrics: Vec::new(),
        }
    }

    pub fn count(&mut self, n: usize) {
        self.total += n;
    }

    pub fn add<T: Metric>(&mut self, metric: T) {
        let id = std::any::TypeId::of::<T>();
        for (t, m) in &mut self.metrics {
            if id == *t {
                m.merge(&metric);
                return;
            }
        }
        self.metrics.push((id, Box::new(metric)));
    }

    pub fn display_metrics(&self) -> String {
        let mut s = String::new();
        for (_, m) in &self.metrics {
            s += &format!("{}: {:.4}, ", m.name(), m.value() / self.total as f32);
        }
        s
    }

    pub fn merge(&mut self, other: Self) {
        self.total += other.total;
        'outer: for (t2, m2) in other.metrics {
            for (t, m) in &mut self.metrics {
                if *t == t2 {
                    m.merge(m2.as_ref());
                    continue 'outer;
                }
            }
            self.metrics.push((t2, m2));
        }
    }

    pub fn len(&self) -> usize {
        self.metrics.len()
    }
}
