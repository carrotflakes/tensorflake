use crate::{Function, Variable};

pub struct MatTranspose;

impl Function for MatTranspose {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<crate::Tensor> {
        assert!(xs.len() == 1);
        assert!(xs[0].shape().len() >= 2);

        let mut axes: Vec<_> = (0..xs[0].shape().len()).collect();
        axes[xs[0].shape().len() - 2..].reverse();

        vec![xs[0].view().permuted_axes(axes).into_owned()]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        MatTranspose.call(vec![gys[0].clone()])
    }
}

#[test]
fn test() {
    use crate::{call, Variable, ENABLE_BACKPROP};

    {
        let x = Variable::<ENABLE_BACKPROP>::new(ndarray::Array::zeros([1, 2, 3]).into_dyn());
        let y = call!(MatTranspose, x);
        assert_eq!(y.shape(), &[1, 3, 2]);

        y.set_grad(Variable::<ENABLE_BACKPROP>::new(
            ndarray::Array::zeros([1, 3, 2]).into_dyn(),
        ));
        y.backward(false, false);

        assert_eq!(x.get_grad::<ENABLE_BACKPROP>().unwrap().shape(), &[1, 2, 3]);
    }
}
