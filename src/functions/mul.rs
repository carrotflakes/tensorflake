use crate::*;

use super::{sum_axes_to_desire, Sum};

pub struct Mul;

impl Function for Mul {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
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
        vec![y.into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
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
                    g = call!(Sum::new(sum_axes_to_desire(g.shape(), x.shape()), false), g);
                }

                g
            })
            .collect()
    }
}

pub fn broadcasted_shape(xs: &[impl std::ops::Deref<Target = Tensor>]) -> Option<Vec<usize>> {
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
        &ndarray::Array::zeros([1, 1, 1]).into_tensor(),
        &ndarray::Array::zeros([1, 1, 2]).into_tensor(),
        &ndarray::Array::zeros([3, 1, 1]).into_tensor(),
    ]);
    assert_eq!(s, Some(vec![3, 1, 2]));

    let s = broadcasted_shape(&[
        &ndarray::Array::zeros([1, 4, 1]).into_tensor(),
        &ndarray::Array::zeros([3, 4, 2]).into_tensor(),
        &ndarray::Array::zeros([1, 4, 2]).into_tensor(),
    ]);
    assert_eq!(s, Some(vec![3, 4, 2]));
}
