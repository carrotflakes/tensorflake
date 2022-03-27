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
