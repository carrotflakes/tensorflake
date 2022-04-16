use crate::{functions::select, initializers::Initializer, *};

pub struct Embedding {
    pub embedding_size: usize,
    pub weights: Param,
}

impl Embedding {
    pub fn new(embedding_size: usize, len: usize, init: &mut impl Initializer) -> Self {
        Self {
            embedding_size,
            weights: init.initialize(&[len, embedding_size]),
        }
    }
}

impl Layer for Embedding {
    type Input = Vec<usize>;
    type Output = Tensor;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        let w = self.weights.get_tensor();
        // Slices::new(x.iter().map(|i| s![*i, ..]).collect())
        //     .call(vec![w])
        select(0, x, &w)
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.weights.clone()]
    }
}