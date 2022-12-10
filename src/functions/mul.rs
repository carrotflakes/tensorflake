use crate::{functions::sum, *};

use super::sum_axes_to_desire;

pub fn mul(a: &ComputedNDA, b: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((&**a * &**b).into_ndarray());

    chain(
        &[a.clone(), b.clone()],
        &[y.clone()],
        false,
        "mul",
        |xs, _ys, gys| {
            let mut gx0 = &gys[0] * &xs[1];
            let mut gx1 = &gys[0] * &xs[0];

            // fit shape
            if xs[0].shape() != gx0.shape() {
                gx0 = gx0.sum(sum_axes_to_desire(gx0.shape(), xs[0].shape()), false);
            }

            if xs[1].shape() != gx1.shape() {
                gx1 = gx1.sum(sum_axes_to_desire(gx1.shape(), xs[1].shape()), false);
            }

            vec![gx0, gx1]
        },
    );

    y
}

pub fn multi_mul(xs: &[ComputedNDA]) -> ComputedNDA {
    assert!(xs.len() >= 1);

    // NOTE: This assert is unnecessary?
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
    let y = ComputedNDA::new(y.into_ndarray());

    chain(xs, &[y.clone()], false, "multi_mul", |xs, _ys, gys| {
        xs.iter()
            .enumerate()
            .map(|(i, x)| {
                let mut g = multi_mul(
                    &(0..xs.len())
                        .filter(|j| *j != i)
                        .map(|j| xs[j].clone())
                        .chain(gys.iter().cloned())
                        .collect::<Vec<_>>(),
                );

                // fit shape
                if x.shape() != g.shape() {
                    g = sum(&g, sum_axes_to_desire(g.shape(), x.shape()), false);
                    // TODO: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/06419d7fb2e7ea19aa3719efc27795edbdc41a1f/dezero/utils.py#L125
                }

                g
            })
            .collect()
    });

    y
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
