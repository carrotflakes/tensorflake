use crate::*;

pub struct CreateGraph {
    y: NDArray,
}

impl CreateGraph {
    pub fn new(y: NDArray) -> Self {
        Self { y }
    }
}

impl Function for CreateGraph {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        #![allow(unused_variables)]

        vec![Tensor::new(self.y.clone())]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        #![allow(unused_variables)]

        vec![]
    }

    const IS_FORCE_CREATE_GRAPH: bool = true;
}
