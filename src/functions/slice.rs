use std::ops::AddAssign;

use ndarray::{IxDyn, SliceArg};

use crate::*;

pub fn slice<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static>(
    x: &ComputedNDA,
    slice_arg: I,
) -> ComputedNDA {
    let y = (&**x).slice(slice_arg.clone());
    let y = ComputedNDA::new(y.into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "slice",
        move |xs, ys, gys| {
            let x = &*xs[0];
            let mut gx = NDArray::zeros(x.shape()); // TODO: Too large tensor!
            gx.slice_mut(slice_arg.clone())
                .assign(&(*gys[0]).reshape(ys[0].shape()));
            vec![ComputedNDA::new(gx)]
        },
    );

    y
    // Slice::new(slice_arg.clone()).forward(&[x.clone()]).pop().unwrap()
}

pub fn slices<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static>(
    x: &ComputedNDA,
    slice_args: Vec<I>,
) -> Vec<ComputedNDA> {
    let xa = &**x;
    let ys: Vec<_> = slice_args
        .iter()
        .map(|slice_arg| xa.slice(slice_arg.clone()).into_ndarray().into())
        .collect();

    chain(&[x.clone()], &ys, false, "slices", move |xs, ys, gys| {
        let x = &*xs[0];
        let mut gx = NDArray::zeros(x.shape());
        for i in 0..xs.len() {
            gx.slice_mut(slice_args[i].clone())
                .add_assign(&(*gys[i]).reshape(ys[i].shape()));
        }
        vec![ComputedNDA::new(gx)]
    });

    ys
}

// TODO:
// #[test]
// fn test_slices() {
//     let x = backprop(NDArray::zeros(&[2, 3][..]));
//     let ys = x.slices(vec![ndarray::s![0, ..], ndarray::s![1, ..]]);
//     let y0 = ys[0].clone();
//     drop(ys);
//     optimize(&y0);
// }
