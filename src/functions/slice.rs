use std::ops::AddAssign;

use ndarray::{IxDyn, SliceArg};

use crate::*;

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
