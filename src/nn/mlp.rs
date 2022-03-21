use ndarray_rand::rand::Rng;

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
        param_gen: &(impl Fn(Tensor) -> Box<dyn Fn() -> Variable> + 'static),
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            linears: sizes
                .windows(2)
                .map(|w| Linear::new(w[0], w[1], param_gen, rng))
                .collect(),
            dropout,
            activation: Box::new(activation),
        }
    }
}

impl Layer for MLP {
    fn call(&self, xs: Vec<Variable>) -> Vec<Variable>
    where
        Self: Sized + 'static,
    {
        let mut ys = xs.clone();
        for linear in &self.linears[..self.linears.len() - 1] {
            ys = linear.call(ys);
            ys = (self.activation)(ys);
            if let Some(rate) = self.dropout {
                ys = Dropout::new(rate).call(ys);
            }
        }
        self.linears.last().unwrap().call(ys)
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
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let mlp = MLP::new(
        &[2, 3, 1],
        None,
        |xs| Sigmoid.call(xs),
        &|x| {
            let o = crate::optimizees::MomentumSGDOptimizee::new(x, 0.9);
            Box::new(move || o.get())
        },
        &mut rng,
    );

    let x = Variable::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_tensor());

    let y = call!(mlp, x);
    // dbg!(&*y);
    assert_eq!(y.shape(), &[4, 1]);
}
