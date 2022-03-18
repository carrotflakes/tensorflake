use crate::*;

pub struct MatTranspose;

impl Function for MatTranspose {
    fn forward(
        &self,
        xs: &Vec<Variable>,
    ) -> Vec<crate::Tensor> {
        assert!(xs.len() == 1);
        assert!(xs[0].shape().len() >= 2);

        let mut axes: Vec<_> = (0..xs[0].shape().len()).collect();
        axes[xs[0].shape().len() - 2..].reverse();

        vec![xs[0].view().permuted_axes(axes).into_tensor()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        MatTranspose.call(vec![gys[0].clone()])
    }
}

// #[test]
// fn test() {
//     use crate::{call, Variable, ENABLE_BACKPROP};

//     {
//         let x = Variable::new(ndarray::Array::zeros([1, 2, 3]).into_dyn());
//         let y = call!(MatTranspose, x);
//         assert_eq!(y.shape(), &[1, 3, 2]);

//         y.backward(false, false);

//         assert_eq!(x.get_grad().unwrap().shape(), &[1, 2, 3]);
//     }
// }
