use crate::*;

use super::SumTo;

pub struct BroadcastTo {
    pub shape: Vec<usize>,
    axes: Vec<usize>,
}

impl BroadcastTo {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            axes: Vec::new(),
        }
    }
}

impl Function for BroadcastTo {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![Variable::new(
            xs[0]
                .broadcast(self.shape.as_slice())
                .unwrap_or_else(|| {
                    panic!("illegal broadcast: {:?} to {:?}", xs[0].shape(), self.shape)
                })
                .into_tensor(),
        )]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        vec![call!(SumTo::new(self.axes.clone()), &gys[0])]
    }

    fn into_backward(mut self, xs: &Vec<Variable>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        // TODO: test
        let axes = &mut self.axes;
        let mut target = xs[0].shape().to_vec();
        for (axis, size) in self.shape.iter().enumerate() {
            if let Some(s) = target.first() {
                if s == size {
                    target.remove(0);
                    continue;
                }
            }
            axes.push(axis);
        }
        Box::new(self)
    }
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_tensor());
        let ys = BroadcastTo::new(vec![2, 3]).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[2, 3]);
        assert_eq!(&*ys[0], &*x);
    }

    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_tensor());
        let ys = BroadcastTo::new(vec![4, 2, 3]).call(vec![x.clone()]);
        // dbg!(&*ys[0]);
        assert_eq!(ys[0].shape(), &[4, 2, 3]);
    }
}
