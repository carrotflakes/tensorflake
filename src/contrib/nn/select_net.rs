use crate::{functions::*, ndarray_util::as_2d, *};

use ndarray::s;

pub struct SelectNet<
    S: Layer<Input = ComputedNDA, Output = ComputedNDA>,
    L: Layer<Input = ComputedNDA, Output = ComputedNDA>,
> {
    pub output_size: usize,
    pub select_layer: S,
    pub layers: Vec<L>,
}

impl<
        S: Layer<Input = ComputedNDA, Output = ComputedNDA>,
        L: Layer<Input = ComputedNDA, Output = ComputedNDA>,
    > SelectNet<S, L>
{
    pub fn new(
        input: usize,
        output: usize,
        n: usize,
        select_layer_builder: impl Fn(usize, usize) -> S,
        layer_builder: impl Fn(usize, usize, usize) -> L,
    ) -> Self {
        Self {
            output_size: output,
            select_layer: select_layer_builder(input, n),
            layers: (0..n).map(|i| layer_builder(i, input, output)).collect(),
        }
    }

    pub fn _call(&self, x: ComputedNDA, train: bool) -> (ComputedNDA, ComputedNDA) {
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
                let ly = layer.call(
                    ComputedNDA::new(x.slice(s![i..=i, ..]).into_ndarray()),
                    train,
                );
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

impl<
        S: Layer<Input = ComputedNDA, Output = ComputedNDA>,
        L: Layer<Input = ComputedNDA, Output = ComputedNDA>,
    > Layer for SelectNet<S, L>
{
    type Input = ComputedNDA;
    type Output = ComputedNDA;

    fn call(&self, input: Self::Input, train: bool) -> Self::Output {
        self._call(input, train).0
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        self.select_layer
            .all_params()
            .into_iter()
            .chain(self.layers.iter().flat_map(|l| l.all_params()))
            .collect()
    }
}

#[test]
fn test() {
    use initializers::Scope;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use nn::Linear;

    let init = initializers::with_optimizer::InitializerWithOptimizer::new(
        initializers::random_initializer::RandomInitializer::new(Uniform::new(-0.01, 0.01)),
        optimizers::Adam::new(),
    );

    let select_net = SelectNet::new(
        2,
        3,
        10,
        |i, o| {
            Linear::new(
                i,
                o,
                init.scope("select_net_select_w"),
                Some(init.scope("select_net_select_b")),
            )
        },
        |n, i, o| {
            Linear::new(
                i,
                o,
                init.scope(format!("select_net_layer_w_{}", n)),
                Some(init.scope(format!("select_net_layer_b_{}", n))),
            )
        },
    );

    let x = backprop(array![[0.1, 0.2], [0.0, 0.0], [0.0, 100.0]].into_ndarray());
    let y = select_net._call(x.clone(), true);
    dbg!(&*y.0);
    // dbg!(&*y[1]);
    let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_ndarray()
        .into();
    let loss = losses::naive_mean_squared_error(y.0.clone(), t);
    dbg!(loss[[]]);
    // export_dot::export_dot(&[loss.clone()], &format!("select_net.dot")).unwrap();
    optimize(&loss);

    let y2 = select_net._call(x.clone(), true);
    assert_ne!(&*y.0, &*y2.0);
}
