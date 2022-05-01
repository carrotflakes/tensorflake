use super::*;
use crate::{initializers::Initializer, *};

pub struct MLP {
    pub linears: Vec<Linear>,
    pub dropout: Option<Dropout>,
    pub activation: Box<dyn Fn(Computed) -> Computed + Sync + Send>,
}

impl MLP {
    pub fn new(
        sizes: &[usize],
        dropout: Option<Dropout>,
        activation: impl Fn(Computed) -> Computed + Sync + Send + 'static,
        w: impl Initializer,
        b: impl Initializer,
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
    type Input = Computed;
    type Output = Computed;

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

    fn all_params(&self) -> Vec<Param> {
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

    let init = initializers::InitializerWithOptimizer::new(
        Uniform::new(-0.01, 0.01),
        optimizers::Adam::new(),
    );

    let mlp = MLP::new(
        &[2, 3, 1],
        None,
        |x| activations::sigmoid(&x),
        init.scope("mlp"),
        init.scope("mlp"),
    );

    let x = Computed::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_ndarray());

    let y = mlp.call(x, true);
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
