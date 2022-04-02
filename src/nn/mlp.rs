use super::*;
use crate::*;

pub struct MLP {
    pub linears: Vec<Linear>,
    pub dropout: Option<Dropout>,
    pub activation: Box<dyn Fn(Tensor) -> Tensor + Sync + Send>,
}

impl MLP {
    pub fn new(
        sizes: &[usize],
        dropout: Option<Dropout>,
        activation: impl Fn(Tensor) -> Tensor + Sync + Send + 'static,
        w: &mut impl FnMut(&[usize]) -> Param,
        b: &mut impl FnMut(&[usize]) -> Param,
    ) -> Self {
        Self {
            linears: sizes
                .windows(2)
                .map(|x| Linear::new(x[0], x[1], w, b))
                .collect(),
            dropout: dropout,
            activation: Box::new(activation),
        }
    }
}

impl Layer for MLP {
    type Input = Tensor;
    type Output = Tensor;

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
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    let rng = DefaultRng::seed_from_u64(42);

    let param_gen = {
        let rng = rng.clone();
        move || {
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                Param::new(t, optimizers::AdamOptimizer::new())
            }
        }
    };

    let mlp = MLP::new(
        &[2, 3, 1],
        None,
        |x| call!(activations::Sigmoid, x),
        &mut param_gen(),
        &mut param_gen(),
    );

    let x = Tensor::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_ndarray());

    let y = mlp.call(x, true);
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
