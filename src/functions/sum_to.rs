use crate::*;

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
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        let mut x = (*xs[0]).to_owned();
        for axise in self.axes.iter().rev() {
            x = x.sum_axis(ndarray::Axis(*axise));
        }

        vec![x.into_tensor()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        BroadcastTo::new(self.original_shape.clone()).call(vec![gys[0].clone()])
    }

    fn into_backward(mut self, xs: &Vec<Variable>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        self.original_shape = xs[0].shape().to_vec();
        Box::new(self)
    }
}

pub fn sum_to_axes_to_desire(src_shape: &[usize], dst_shape: &[usize]) -> Vec<usize> {
    let mut axes = Vec::new();
    let mut target = dst_shape.to_vec();
    for (axis, size) in src_shape.iter().enumerate() {
        if let Some(s) = target.first() {
            if s == size {
                target.remove(0);
                continue;
            }
        }
        axes.push(axis);
    }
    axes
}

// #[test]
// fn test() {
//     use crate::ENABLE_BACKPROP;

//     {
//         let x = Variable::new(
//             ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
//         );
//         let ys = SumTo::new(vec![0]).call(vec![x.clone()]);
//         assert_eq!(ys[0].shape(), &[3]);
//         assert_eq!(&*ys[0], &ndarray::array![5., 7., 9.].into_dyn());
//     }

//     {
//         let x = Variable::new(
//             ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
//         );
//         let ys = SumTo::new(vec![1]).call(vec![x.clone()]);
//         assert_eq!(ys[0].shape(), &[2]);
//         assert_eq!(&*ys[0], &ndarray::array![6., 15.].into_dyn());
//     }
// }
