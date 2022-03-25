use ndarray::{IxDyn, SliceArg};

use crate::*;

pub struct Slice<I: SliceArg<IxDyn> + Clone + 'static> {
    slice_arg: I,
}

impl<I: SliceArg<IxDyn> + Clone + 'static> Slice<I> {
    pub fn new(slice_arg: I) -> Self {
        Self { slice_arg }
    }
}

impl<I: SliceArg<IxDyn> + Clone + 'static> Function for Slice<I> {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        let y = x.slice(self.slice_arg.clone());
        vec![y.into_ndarray().into()]
    }

    // NOTE: backward cuts the graph.
    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]
        let x = &*xs[0];
        let mut gx = NDArray::zeros(x.shape()); // TODO: Too large tensor!
        gx.slice_mut(self.slice_arg.clone()).assign(&*gys[0]);
        vec![Tensor::new(gx)]
    }
}

// pub struct Slices<const N: usize> {
//     slice_args: Vec<SliceInfo<N>>,
// }

// impl<const N: usize> Slices<N> {
//     pub fn new(slice_args: Vec<SliceInfo<N>>) -> Self {
//         Self { slice_args }
//     }
// }

// impl<const N: usize> Function for Slices<N> {
//     fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
//         assert_eq!(xs.len(), 1);
//         let x = &*xs[0];
//         let y = Tensor::zeros([self.slice_args.len()].into_iter().chain(self.slice_args[0]));
//         let y = &x.slice(self.slice_args.clone());
//         vec![y.into_tensor().into()]
//     }

//     // NOTE: backward cuts the graph
//     fn backward(
//         &self,
//         xs: &Vec<Variable>,
//         ys: &Vec<Variable>,
//         gys: &Vec<Variable>,
//     ) -> Vec<Variable> {
//         #![allow(unused_variables)]
//         let x = &*xs[0];
//         let mut gx = Tensor::zeros(x.shape());
//         gx.slice_mut(self.slice_arg.clone()).assign(&*gys[0]);
//         vec![Variable::new(gx)]
//     }
// }
