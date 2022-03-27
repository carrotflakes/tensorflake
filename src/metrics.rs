use ndarray::s;

use crate::*;

pub fn argmax_accuracy(t: &[usize], y: &Tensor) -> f32 {
    let y = y
        .view()
        .into_shape([
            y.shape().iter().take(y.ndim() - 1).product::<usize>(),
            y.shape()[y.ndim() - 1],
        ])
        .unwrap();

    t.iter()
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
        .count() as f32
        / t.len() as f32
}

pub trait Metric: Sync + Send {
    fn name(&self) -> &'static str;
    fn merge(&mut self, other: &Self)
    where
        Self: Sized;
    fn value(&self) -> f32;
}

pub struct Loss(f32);

impl Loss {
    pub fn new(loss: f32) -> Loss {
        Loss(loss)
    }
}

impl Metric for Loss {
    fn name(&self) -> &'static str {
        "loss"
    }

    fn merge(&mut self, other: &Self) {
        self.0 += other.0;
    }

    fn value(&self) -> f32 {
        self.0 as f32
    }
}

pub struct Accuracy(usize);

impl Accuracy {
    pub fn new(correct: usize) -> Accuracy {
        Accuracy(correct)
    }
}

impl Metric for Accuracy {
    fn name(&self) -> &'static str {
        "accuracy"
    }

    fn merge(&mut self, other: &Self) {
        self.0 += other.0;
    }

    fn value(&self) -> f32 {
        self.0 as f32
    }
}
