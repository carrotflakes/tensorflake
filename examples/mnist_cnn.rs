mod data;

use ndarray::prelude::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Normal, RandomExt};
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    nn::{activations::Relu, naive_max_pooling, Conv2d, Dropout, Layer, Linear},
    training::{TrainConfig, UpdateStrategy},
    *,
};

fn main() {
    let mnist = data::mnist::Mnist::load("./data/mnist");

    let model = Model::new();
    param_bin::params_summary(&model.all_params());

    let start = std::time::Instant::now();

    TrainConfig {
        epoch: 100,
        train_data: mnist.trains().collect(),
        validation_data: mnist.tests().collect(),
        validation_rate: 0.1,
        batch_size: 100,
        parallel: true,
        update_strategy: UpdateStrategy::MiniBatch(1),
        ..Default::default()
    }
    .build()
    .fit(|batch, ctx| {
        let x = NDArray::from_shape_vec(
            &[batch.len(), 1, 28, 28][..],
            batch
                .iter()
                .flat_map(|x| x.0)
                .map(|x| *x as f32 / 255.0)
                .collect(),
        )
        .unwrap();
        let t: Vec<_> = batch.iter().map(|x| x.1 as usize).collect();
        let y = model.call(x.clone(), ctx.train);
        let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
        ctx.finish_batch(&loss, t.len());
        ctx.add_metric(metrics::argmax_accuracy(&t, &y));
    });

    println!("time: {:?}", start.elapsed());
}

pub struct Model {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub linear: Linear,
}

impl Model {
    pub fn new() -> Self {
        let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
        let param_gen = {
            let rng = rng.clone();
            move || {
                let mut rng = rng.clone();
                move |shape: &[usize]| -> Param {
                    let t = Array::random_using(shape, Normal::new(0.0, 0.1).unwrap(), &mut rng)
                        .into_ndarray();
                    Param::new(t, optimizers::AdamOptimizer::new())
                }
            }
        };
        Self {
            conv1: Conv2d::new(
                [3, 3],
                // [1, 1],
                [2, 2],
                [1, 1],
                param_gen()(&[10, 1, 3, 3]),
                Some(param_gen()(&[10])),
            ),
            conv2: Conv2d::new(
                [3, 3],
                [2, 2],
                [1, 1],
                param_gen()(&[10, 10, 3, 3]),
                Some(param_gen()(&[10])),
            ),
            linear: Linear::new(10 * 7 * 7, 10, &mut param_gen(), &mut param_gen()),
        }
    }

    pub fn call(&self, x: NDArray, train: bool) -> Tensor {
        let y = self.conv1.call(Tensor::new(x), train);
        // let y = naive_max_pooling(&y, [2, 2], [2, 2], [0, 0]);
        let y = call!(Relu, y);
        let y = self.conv2.call(y, train);
        let y = call!(Relu, y);
        let y = y.reshape([y.shape()[0], 10 * 7 * 7]);
        let y = self.linear.call(y, train);
        y
    }

    pub fn all_params(&self) -> Vec<Param> {
        self.conv1
            .all_params()
            .into_iter()
            .chain(self.conv2.all_params())
            .chain(self.linear.all_params())
            .collect()
    }
}
pub struct BigModel {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub linear1: Linear,
    pub linear2: Linear,
    pub dropout: Dropout,
}

impl BigModel {
    pub fn new() -> Self {
        let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
        let param_gen = {
            let rng = rng.clone();
            move || {
                let mut rng = rng.clone();
                move |shape: &[usize]| -> Param {
                    let t = Array::random_using(shape, Normal::new(0.0, 0.1).unwrap(), &mut rng)
                        .into_ndarray();
                    Param::new(t, optimizers::AdamOptimizer::new())
                }
            }
        };
        Self {
            conv1: Conv2d::new(
                [3, 3],
                [1, 1],
                [0, 0],
                param_gen()(&[32, 1, 3, 3]),
                Some(param_gen()(&[32])),
            ),
            conv2: Conv2d::new(
                [3, 3],
                [1, 1],
                [0, 0],
                param_gen()(&[64, 32, 3, 3]),
                Some(param_gen()(&[64])),
            ),
            linear1: Linear::new(64 * 12 * 12, 128, &mut param_gen(), &mut param_gen()),
            linear2: Linear::new(128, 10, &mut param_gen(), &mut param_gen()),
            dropout: Dropout::new(0.5, 42),
        }
    }

    pub fn call(&self, x: NDArray, train: bool) -> Tensor {
        let x = Tensor::new(x);
        let y = self.conv1.call(x, train);
        let y = call!(Relu, y);
        let y = self.conv2.call(y, train);
        let y = naive_max_pooling(&y, [2, 2], [2, 2], [0, 0]);
        let y = y.reshape([y.shape()[0], 64 * 12 * 12]);
        let y = call!(Relu, y);
        let y = self.linear1.call(y, train);
        let y = call!(Relu, y);
        let y = self.dropout.call(y, train);
        let y = self.linear2.call(y, train);
        y
    }

    pub fn all_params(&self) -> Vec<Param> {
        self.conv1
            .all_params()
            .into_iter()
            .chain(self.conv2.all_params())
            .chain(self.linear1.all_params())
            .chain(self.linear2.all_params())
            .collect()
    }
}
