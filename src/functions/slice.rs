use std::ops::AddAssign;

use ndarray::{IxDyn, SliceArg};

use crate::*;

pub fn slice<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static>(
    x: &Tensor,
    slice_arg: I,
) -> Tensor {
    let y = (&**x).slice(slice_arg.clone());
    let y = Tensor::new(y.into_ndarray());

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
            vec![Tensor::new(gx)]
        },
    );

    y
    // Slice::new(slice_arg.clone()).forward(&[x.clone()]).pop().unwrap()
}

pub fn slices<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static>(
    x: &Tensor,
    slice_args: Vec<I>,
) -> Vec<Tensor> {
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
        vec![Tensor::new(gx)]
    });

    ys
}

pub struct Slice<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> {
    slice_arg: I,
}

impl<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> Slice<I> {
    pub fn new(slice_arg: I) -> Self {
        Self { slice_arg }
    }
}

impl<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> Function for Slice<I> {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        let y = x.slice(self.slice_arg.clone());
        vec![y.into_ndarray().into()]
    }

    // NOTE: backward cuts the graph.
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]
        let x = &*xs[0];
        let mut gx = NDArray::zeros(x.shape()); // TODO: Too large tensor!
        gx.slice_mut(self.slice_arg.clone())
            .assign(&(*gys[0]).reshape(ys[0].shape()));
        vec![Tensor::new(gx)]
    }
}

pub struct Slices<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> {
    slice_args: Vec<I>,
}

impl<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> Slices<I> {
    pub fn new(slice_args: Vec<I>) -> Self {
        Self { slice_args }
    }
}

impl<I: SliceArg<IxDyn> + Clone + Sync + Send + 'static> Function for Slices<I> {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        self.slice_args
            .iter()
            .map(|slice_arg| x.slice(slice_arg.clone()).into_ndarray().into())
            .collect()
    }

    // NOTE: backward cuts the graph.
    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        let x = &*xs[0];
        let mut gx = NDArray::zeros(x.shape());
        for i in 0..xs.len() {
            gx.slice_mut(self.slice_args[i].clone())
                .add_assign(&(*gys[i]).reshape(ys[i].shape()));
        }
        vec![Tensor::new(gx)]
    }
}
