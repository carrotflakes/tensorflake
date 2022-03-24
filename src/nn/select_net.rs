use crate::{functions::*, tensor_util::as_2d, *};

use ndarray::s;

use super::{Linear, Softmax};

pub struct SelectNet {
    pub output_size: usize,
    pub select_layer: Linear,
    pub layers: Vec<Linear>,
}

impl SelectNet {
    pub fn new(
        input: usize,
        output: usize,
        n: usize,
        w: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
        b: &mut impl FnMut(&[usize]) -> Box<dyn Fn() -> Variable>,
    ) -> Self {
        Self {
            output_size: output,
            select_layer: Linear::new(input, n, w, b),
            layers: (0..n)
                .map(move |_| Linear::new(input, output, w, b))
                .collect(),
        }
    }

    pub fn build(&self) -> Self {
        let select_layer = self.select_layer.build();
        let layers = self.layers.iter().map(|layer| layer.build()).collect();
        Self {
            output_size: self.output_size,
            select_layer,
            layers,
        }
    }

    pub fn call(&self, xs: Vec<Variable>, train: bool) -> Vec<Variable> {
        let select = self.select_layer.call(xs.to_vec(), train).pop().unwrap();
        let x = as_2d(&xs[0]);
        let softmax = call!(Softmax, select);

        let mut ys = vec![];
        for i in 0..x.shape()[0] {
            let mut select = (&*softmax)
                .slice(s![i, ..].to_owned())
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            select.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            select.truncate(5);

            let mut lys = Vec::new();
            for (j, _) in &select {
                let layer = &self.layers[*j];
                let ly = layer
                    .call(
                        vec![Variable::new(x.slice(s![i..=i, ..]).into_tensor())],
                        train,
                    )
                    .pop()
                    .unwrap();
                lys.push(call!(Mul, ly, call!(Slice::new(s![i, *j]), softmax)));
            }
            let lys = Add.call(lys).pop().unwrap();
            ys.push(lys);
        }

        let y = Concat::new(1).call(ys).pop().unwrap();
        // reshape to original shape
        let y = call!(
            Reshape::new(
                xs[0]
                    .shape()
                    .iter()
                    .take(xs[0].ndim() - 1)
                    .chain([self.output_size].iter())
                    .copied()
                    .collect::<Vec<_>>()
            ),
            y
        );

        vec![y, softmax]
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

    let select_net = SelectNet::new(2, 3, 10, &mut param_gen(), &mut param_gen());

    let x = backprop(array![[0.1, 0.2], [0.0, 0.0], [0.0, 100.0]].into_tensor());
    let y = select_net.build().call(vec![x.clone()], true);
    dbg!(&*y[0]);
    // dbg!(&*y[1]);
    let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_tensor()
        .into();
    let loss = losses::naive_mean_squared_error(y[0].clone(), t);
    dbg!(loss[[]]);
    // export_dot::export_dot(&[loss.clone()], &format!("select_net.dot")).unwrap();
    optimize(&loss, 0.1);

    let y = select_net.build().call(vec![x.clone()], true);
    dbg!(&*y[0]);
}
