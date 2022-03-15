use crate::{Backward, Function, Tensor, Variable};

use super::BroadcastTo;

pub struct SumTo {
    // NOTE: axes are in order
    pub axes: Vec<usize>,
    original_shape: Vec<usize>,
}

impl SumTo {
    pub fn new(axes: Vec<usize>) -> Self {
        assert!(axes.windows(2).all(|w| w[0] < w[1]));
        Self {
            axes,
            original_shape: Vec::new(),
        }
    }
}

impl Function for SumTo {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        let mut x = (*xs[0]).to_owned();
        for axise in self.axes.iter().rev() {
            x = x.sum_axis(ndarray::Axis(*axise));
        }

        vec![x]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        BroadcastTo::new(self.original_shape.clone()).call(vec![gys[0].clone()])
    }

    fn into_backward(mut self, xs: &Vec<Variable<true>>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        self.original_shape = xs[0].shape().to_vec();
        Box::new(self)
    }
}

#[test]
fn test() {
    use crate::ENABLE_BACKPROP;

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let ys = SumTo::new(vec![0]).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[3]);
        assert_eq!(&*ys[0], &ndarray::array![5., 7., 9.].into_dyn());
    }

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let ys = SumTo::new(vec![1]).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[2]);
        assert_eq!(&*ys[0], &ndarray::array![6., 15.].into_dyn());
    }
}
