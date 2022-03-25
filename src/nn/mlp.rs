use super::*;
use crate::*;

pub struct MLP {
    pub linears: Vec<Linear>,
    pub dropout: Option<f32>,
    pub activation: Box<dyn Fn(Vec<Tensor>) -> Vec<Tensor>>,
}

impl MLP {
    pub fn new(
        sizes: &[usize],
        dropout: Option<f32>,
        activation: impl Fn(Vec<Tensor>) -> Vec<Tensor> + 'static,
        w: &mut impl FnMut(&[usize]) -> Param,
        b: &mut impl FnMut(&[usize]) -> Param,
    ) -> Self {
        Self {
            linears: sizes
                .windows(2)
                .map(|x| Linear::new(x[0], x[1], w, b))
                .collect(),
            dropout,
            activation: Box::new(activation),
        }
    }
}

impl Layer for MLP {
    fn call(&self, xs: Vec<Tensor>, train: bool) -> Vec<Tensor>
    where
        Self: Sized + 'static,
    {
        let mut ys = xs.clone();
        for linear in &self.linears[..self.linears.len() - 1] {
            ys = linear.call(ys, train);
            ys = (self.activation)(ys);
            if let Some(rate) = self.dropout {
                ys = Dropout::new(rate).call(ys, train);
            }
        }
        self.linears.last().unwrap().call(ys, train)
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
    let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    let param_gen = {
        let rng = rng.clone();
        move || {
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                AdamOptimizee::new(t)
            }
        }
    };

    let mlp = MLP::new(
        &[2, 3, 1],
        None,
        |xs| Sigmoid.call(xs),
        &mut param_gen(),
        &mut param_gen(),
    );

    let x = Tensor::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_ndarray());

    let y = mlp.call(vec![x], true).pop().unwrap();
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
