use crate::{functions::*, ndarray_util::as_2d, *};

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
        w: &mut impl FnMut(&[usize]) -> Param,
        b: &mut impl FnMut(&[usize]) -> Param,
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

    pub fn call(&self, x: Tensor, train: bool) -> (Tensor, Tensor) {
        let select = self.select_layer.call(x.clone(), train);
        let x = as_2d(&x);
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
                let ly = layer.call(Tensor::new(x.slice(s![i..=i, ..]).into_ndarray()), train);
                lys.push(call!(Mul, ly, call!(Slice::new(s![i, *j]), softmax)));
            }
            let lys = Add.call(lys).pop().unwrap();
            ys.push(lys);
        }

        let y = Concat::new(1).call(ys).pop().unwrap();
        // reshape to original shape
        let y = call!(
            Reshape::new(
                x.shape()
                    .iter()
                    .take(x.ndim() - 1)
                    .chain([self.output_size].iter())
                    .copied()
                    .collect::<Vec<_>>()
            ),
            y
        );

        (y, softmax)
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

    let select_net = SelectNet::new(2, 3, 10, &mut param_gen(), &mut param_gen());

    let x = backprop(array![[0.1, 0.2], [0.0, 0.0], [0.0, 100.0]].into_ndarray());
    let y = select_net.build().call(x.clone(), true);
    dbg!(&*y.0);
    // dbg!(&*y[1]);
    let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_ndarray()
        .into();
    let loss = losses::naive_mean_squared_error(y.0.clone(), t);
    dbg!(loss[[]]);
    // export_dot::export_dot(&[loss.clone()], &format!("select_net.dot")).unwrap();
    optimize(&loss, 0.1);

    let y = select_net.build().call(x.clone(), true);
    dbg!(&*y.0);
}
