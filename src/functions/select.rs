use std::ops::AddAssign;

use ndarray::Axis;

use crate::*;

pub fn select(axis: usize, indices: Vec<usize>, x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new(x.select(Axis(axis), &indices).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "select",
        move |xs, ys, gys| {
            let _ = ys;
            let mut gx = NDArray::zeros(xs[0].shape());
            for i in 0..indices.len() {
                gx.index_axis_mut(Axis(axis), indices[i])
                    .add_assign(&gys[0].index_axis(Axis(0), i));
            }
            vec![ComputedNDA::new(gx)]
        },
    );

    y
}
