use crate::*;

pub struct Relu;

impl Function for Relu {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.max(0.0)).into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        vec![call!(
            functions::Mul,
            gys[0],
            Tensor::new(
                xs[0]
                    .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                    .into_ndarray()
            )
        )]
    }
}
