use crate::{Function, Variable};

pub struct T;

impl Function for T {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<crate::Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].t().into_owned()]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        T.call(vec![gys[0].clone()])
    }
}

#[test]
fn test() {
    use crate::{Variable, ENABLE_BACKPROP};

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
        );
        let ys = T.call(vec![x.clone()]);
        assert_eq!(&ys[0].shape(), &[3, 2]);
    }

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_dyn(),
        );
        let ys = T.call(vec![x.clone()]);
        assert_eq!(&ys[0].shape(), &[3, 2, 1]);
    }
}
