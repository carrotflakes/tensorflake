use crate::*;

use super::{sum_axes_to_desire, Sum};

pub struct Mul;

impl Function for Mul {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        if broadcasted_shape(&xs).is_none() {
            panic!(
                "cannot broadcast on shapes: {:?}",
                xs.iter()
                    .map(|x| (**x).shape().to_vec())
                    .collect::<Vec<_>>()
            );
        };

        let mut y = (*xs[0]).to_owned();
        for x in xs.iter().skip(1) {
            y = y * &**x;
        }
        vec![y.into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        xs.iter()
            .enumerate()
            .map(|(i, x)| {
                let mut g = Mul
                    .call(
                        (0..xs.len())
                            .filter(|j| *j != i)
                            .map(|j| xs[j].clone())
                            .chain(gys.iter().cloned())
                            .collect(),
                    )
                    .pop()
                    .unwrap();

                // fit shape
                if x.shape() != g.shape() {
                    g = call!(Sum::new(sum_axes_to_desire(g.shape(), x.shape()), true), g);
                    // TODO: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/06419d7fb2e7ea19aa3719efc27795edbdc41a1f/dezero/utils.py#L125
                }

                g
            })
            .collect()
    }
}

pub fn broadcasted_shape(xs: &[impl std::ops::Deref<Target = NDArray>]) -> Option<Vec<usize>> {
    let mut shape = xs[0].shape().to_vec();
    for x in xs.iter().skip(1) {
        let x_shape = x.shape();

        // scalar is broadcasted to any shape
        if shape.len() == 0 {
            shape = x_shape.to_vec();
            continue;
        }
        if x_shape.len() == 0 {
            continue;
        }

        if shape.len() != x_shape.len() {
            return None;
        }

        for i in 0..shape.len() {
            match (shape[i], x_shape[i]) {
                (1, _) => shape[i] = x_shape[i],
                (_, 1) => (),
                (s1, s2) if s1 == s2 => (),
                (_, _) => return None,
            }
        }
    }
    Some(shape)
}

#[test]
fn test() {
    let s = broadcasted_shape(&[
        &ndarray::Array::zeros([1, 1, 1]).into_ndarray(),
        &ndarray::Array::zeros([1, 1, 2]).into_ndarray(),
        &ndarray::Array::zeros([3, 1, 1]).into_ndarray(),
    ]);
    assert_eq!(s, Some(vec![3, 1, 2]));

    let s = broadcasted_shape(&[
        &ndarray::Array::zeros([1, 4, 1]).into_ndarray(),
        &ndarray::Array::zeros([3, 4, 2]).into_ndarray(),
        &ndarray::Array::zeros([1, 4, 2]).into_ndarray(),
    ]);
    assert_eq!(s, Some(vec![3, 4, 2]));
}
