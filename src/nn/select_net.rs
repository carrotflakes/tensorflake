use crate::{functions::*, initializers::Initializer, ndarray_util::as_2d, *};

use ndarray::s;

use super::Linear;

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
        w: impl Initializer,
        b: impl Initializer,
    ) -> Self {
        Self {
            output_size: output,
            select_layer: Linear::new(
                input,
                n,
                w.scope("select_layer"),
                Some(b.scope("select_layer")),
            ),
            layers: (0..n)
                .map(|i| {
                    Linear::new(
                        input,
                        output,
                        w.scope(format!("layer_{}", i)),
                        Some(b.scope(format!("layer_{}", i))),
                    )
                })
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
        let softmax = nn::activations::softmax(&select);

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
                lys.push(ly * softmax.slice(s![i, *j]));
            }
            let lys = multi_add(&lys);
            ys.push(lys);
        }

        let y = concat(&ys, 1);
        // reshape to original shape
        let y = y.reshape(
            x.shape()
                .iter()
                .take(x.ndim() - 1)
                .chain([self.output_size].iter())
                .copied()
                .collect::<Vec<_>>(),
        );

        (y, softmax)
    }
}

#[test]
fn test() {
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;

    let init = initializers::InitializerWithOptimizer::new(
        Uniform::new(-0.01, 0.01),
        optimizers::AdamOptimizer::new(),
    );

    let select_net = SelectNet::new(2, 3, 10, init.scope("select_net"), init.scope("select_net"));

    let x = backprop(array![[0.1, 0.2], [0.0, 0.0], [0.0, 100.0]].into_ndarray());
    let y = select_net.call(x.clone(), true);
    dbg!(&*y.0);
    // dbg!(&*y[1]);
    let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_ndarray()
        .into();
    let loss = losses::naive_mean_squared_error(y.0.clone(), t);
    dbg!(loss[[]]);
    // export_dot::export_dot(&[loss.clone()], &format!("select_net.dot")).unwrap();
    optimize(&loss);

    let y2 = select_net.build().call(x.clone(), true);
    assert_ne!(&*y.0, &*y2.0);
}
