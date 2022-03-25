use crate::*;

pub struct MatTranspose;

impl Function for MatTranspose {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        assert!(xs[0].shape().len() >= 2);

        vec![forward(&xs[0]).into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        MatTranspose.call(vec![gys[0].clone()])
    }
}

pub fn forward(x: &NDArray) -> NDArray {
    let mut axes: Vec<_> = (0..x.shape().len()).collect();
    axes[x.shape().len() - 2..].reverse();

    x.view().permuted_axes(axes).into_ndarray()
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::Array::zeros([1, 2, 3]).into_ndarray());
        let y = call!(MatTranspose, x);
        assert_eq!(y.shape(), &[1, 3, 2]);

        let grads = gradients(&[y.clone()], &[x.clone()], false);
        assert_eq!(grads[0].shape(), &[1, 2, 3]);
    }
}
