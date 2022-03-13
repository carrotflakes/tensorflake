use crate::{Backward, Function, Tensor, Variable};

use super::SumTo;

pub struct BroadcastTo {
    pub shape: Vec<usize>,
}

impl BroadcastTo {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Function for BroadcastTo {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].broadcast(self.shape.as_slice()).unwrap().into_owned()]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]
        unreachable!()
    }

    fn into_backward(self, xs: &Vec<Variable<true>>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        // TODO: test
        let mut axises = Vec::new();
        let mut target = xs[0].shape().to_vec();
        for (axis, size) in self.shape.iter().enumerate() {
            if let Some(s) = target.first() {
                if s == size {
                    target.remove(0);
                    continue;
                }
            }
            axises.push(axis);
        }
        Box::new(BroadcastToBw {
            broadcasted_shape: self.shape,
            axises,
        })
    }
}

pub struct BroadcastToBw {
    broadcasted_shape: Vec<usize>,
    axises: Vec<usize>,
}

impl Backward for BroadcastToBw {
    fn backward(
        &self,
        xs: &Vec<Variable<true>>,
        gys: &Vec<Variable<true>>,
        enable_backprop: bool,
    ) -> Vec<Variable<true>> {
        #![allow(unused_variables)]

        let gy = gys[0].broadcast(self.broadcasted_shape.as_slice()).unwrap();
        SumTo::new(self.axises.clone()).call(vec![Variable::new(gy.into_owned())])
    }
}

#[test]
fn test() {
    use crate::{scalar, ENABLE_BACKPROP};

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let ys = BroadcastTo::new(vec![2, 3]).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[2, 3]);
        assert_eq!(&*ys[0], &*x);

        ys[0].set_grad(Variable::<ENABLE_BACKPROP>::new(scalar(1.0)));
        ys[0].backward(false, false);
        dbg!(&*x.get_grad::<ENABLE_BACKPROP>().unwrap()); // = [1], are you ok?
    }

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let ys = BroadcastTo::new(vec![4, 2, 3]).call(vec![x.clone()]);
        // dbg!(&*ys[0]);
        assert_eq!(ys[0].shape(), &[4, 2, 3]);

        ys[0].set_grad(Variable::<ENABLE_BACKPROP>::new(scalar(1.0)));
        ys[0].backward(false, false);
        dbg!(&*x.get_grad::<ENABLE_BACKPROP>().unwrap());
    }
}
