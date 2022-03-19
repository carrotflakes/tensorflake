use crate::*;

pub struct Relu;

impl Function for Relu {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.max(0.0)).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        vec![call!(
            functions::Mul,
            gys[0],
            Variable::new(
                xs[0]
                    .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                    .into_tensor()
            )
        )]
    }
}
