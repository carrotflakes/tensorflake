use super::*;
use crate::*;

pub struct MLP {
    pub linears: Vec<Linear>,
    pub dropout: Option<f32>,
    pub activation: Box<dyn Fn(Vec<Variable>) -> Vec<Variable>>,
}

impl MLP {
    pub fn new(
        sizes: &[usize],
        dropout: Option<f32>,
        activation: impl Fn(Vec<Variable>) -> Vec<Variable> + 'static,
        w: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
        b: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
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
    fn call(&self, xs: Vec<Variable>, train: bool) -> Vec<Variable>
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

    fn all_params(&self) -> Vec<Variable> {
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
            move |shape: &[usize]| -> Box<dyn Fn() -> Variable> {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_tensor();
                let o = AdamOptimizee::new(t);
                Box::new(move || o.get())
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

    let x = Variable::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_tensor());

    let y = mlp.call(vec![x], true).pop().unwrap();
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
