use crate::{functions::select, initializers::Initializer, *};

pub struct Embedding {
    pub embedding_size: usize,
    pub weights: ParamNDA,
}

impl Embedding {
    pub fn new(embedding_size: usize, len: usize, init:  impl Initializer<NDArray>) -> Self {
        Self {
            embedding_size,
            weights: init.initialize(&[len, embedding_size]),
        }
    }
}

impl Layer for Embedding {
    type Input = Vec<usize>;
    type Output = ComputedNDA;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        let w = self.weights.get();
        // Slices::new(x.iter().map(|i| s![*i, ..]).collect())
        //     .call(vec![w])
        select(0, x, &w)
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        vec![self.weights.clone()]
    }
}
