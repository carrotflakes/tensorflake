use ndarray::Ix2;

use crate::{Function, Variable};

use super::T;

pub struct Matmul;

impl Function for Matmul {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<crate::Tensor> {
        assert!(xs.len() == 2);

        // 行列同士の積に限定する
        let x0 = (*xs[0]).to_owned().into_dimensionality::<Ix2>().unwrap();
        let x1 = (*xs[1]).to_owned().into_dimensionality::<Ix2>().unwrap();

        vec![x0.dot(&x1).into_dyn()]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        let x = xs[0].clone();
        let w = xs[1].clone();
        let gx = Matmul.call(vec![gys[0].clone(), T.call(vec![w.clone()])[0].clone()])[0].clone();
        let gw = Matmul.call(vec![T.call(vec![x.clone()])[0].clone(), gys[0].clone()])[0].clone();
        vec![gx, gw]
    }
}

#[test]
fn test() {
    use crate::{Variable, ENABLE_BACKPROP};

    {
        let a = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let b = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2.], [3., 4.], [5., 6.]].into_dyn(),
        );
        let ys = Matmul.call(vec![a.clone(), b.clone()]);
        assert_eq!(&ys[0].shape(), &[2, 2]);

        ys[0].set_grad(Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 1.], [1., 1.]].into_dyn(),
        ));
        ys[0].backward(false, false);
        dbg!(&*a.get_grad::<ENABLE_BACKPROP>().unwrap());

        let ys = Matmul.call(vec![b.clone(), a.clone()]);
        assert_eq!(&ys[0].shape(), &[3, 3]);
    }
}
