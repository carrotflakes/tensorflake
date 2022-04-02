use ndarray::Axis;

use crate::*;

use super::Broadcast;

pub struct Sum {
    // NOTE: axes are in order
    pub axes: Vec<usize>,
    pub keep_dim: bool,
    original_shape: Vec<usize>,
}

impl Sum {
    pub fn new(axes: Vec<usize>, keep_dim: bool) -> Self {
        assert!(axes.windows(2).all(|w| w[0] < w[1]));
        Self {
            axes,
            keep_dim,
            original_shape: Vec::new(),
        }
    }
}

impl Function for Sum {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        let mut x = (*xs[0]).to_owned();
        for axis in self.axes.iter().rev() {
            x = x.sum_axis(Axis(*axis));
            if self.keep_dim {
                x.insert_axis_inplace(Axis(*axis));
            }
        }

        vec![x.into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        Broadcast::new(self.original_shape.clone()).call(vec![gys[0].clone()])
    }

    fn into_backward(mut self, xs: &Vec<Tensor>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        self.original_shape = xs[0].shape().to_vec();
        Box::new(self)
    }
}

pub fn sum_axes_to_desire(src_shape: &[usize], dst_shape: &[usize]) -> Vec<usize> {
    assert!(src_shape.len() >= dst_shape.len());
    let offset = src_shape.len() - dst_shape.len();
    let mut axes: Vec<_> = (0..offset).collect();
    for axis in offset..src_shape.len() {
        if dst_shape[axis - offset] == 1 {
            axes.push(axis);
        } else {
            assert!(src_shape[axis] == dst_shape[axis - offset]);
        }
    }
    axes
}

#[test]
fn test_sum_axes_to_desire() {
    assert_eq!(sum_axes_to_desire(&[2, 3, 4], &[2, 1, 4]), vec![1]);
}

#[test]
fn test() {
    {
        let x = Tensor::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let ys = Sum::new(vec![0], false).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[3]);
        assert_eq!(&*ys[0], &ndarray::array![5., 7., 9.].into_ndarray());
    }

    {
        let x = Tensor::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let ys = Sum::new(vec![1], false).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[2]);
        assert_eq!(&*ys[0], &ndarray::array![6., 15.].into_ndarray());
    }
}
