use super::*;
use crate::{
    initializers::{Initializer, Scope},
    *,
};

pub struct MLP {
    pub linears: Vec<Linear>,
    pub dropout: Option<Dropout>,
    pub activation: Box<dyn Fn(ComputedNDA) -> ComputedNDA + Sync + Send>,
}

impl MLP {
    pub fn new(
        sizes: &[usize],
        dropout: Option<Dropout>,
        activation: impl Fn(ComputedNDA) -> ComputedNDA + Sync + Send + 'static,
        w: impl Initializer<ParamNDA> + Scope,
        b: impl Initializer<ParamNDA> + Scope,
    ) -> Self {
        Self {
            linears: sizes
                .windows(2)
                .enumerate()
                .map(|(i, x)| {
                    Linear::new(
                        x[0],
                        x[1],
                        w.scope(format!("linear_{}", i)),
                        Some(b.scope(format!("linear_{}", i))),
                    )
                })
                .collect(),
            dropout,
            activation: Box::new(activation),
        }
    }
}

impl Layer for MLP {
    type Input = ComputedNDA;
    type Output = ComputedNDA;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output {
        let mut y = x.clone();
        for linear in &self.linears[..self.linears.len() - 1] {
            y = linear.call(y, train);
            y = (self.activation)(y);
            if let Some(dropout) = &self.dropout {
                y = dropout.call(y, train);
            }
        }
        self.linears.last().unwrap().call(y, train)
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        self.linears
            .iter()
            .flat_map(|linear| linear.all_params())
            .collect()
    }
}

#[test]
fn test() {
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;

    let init = initializers::with_optimizer::InitializerWithOptimizer::new(
        initializers::random_initializer::RandomInitializer::new(Uniform::new(-0.01, 0.01)),
        optimizers::Adam::new(),
    );

    let mlp = MLP::new(
        &[2, 3, 1],
        None,
        |x| activations::sigmoid(&x),
        init.scope("mlp"),
        init.scope("mlp"),
    );

    let x = ComputedNDA::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_ndarray());

    let y = mlp.call(x, true);
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
