mod data;
mod training;

use ndarray::prelude::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{losses::SoftmaxCrossEntropy, nn::*, *};

use crate::training::TrainingConfig;

fn main() {
    let mnist = data::mnist::Mnist::load("./data");

    let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
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
        &[28 * 28, 128, 10],
        Some(Dropout::new(0.2, 42)),
        |x| Relu.call(vec![x]).pop().unwrap(),
        &mut param_gen(),
        &mut param_gen(),
    );

    let start = std::time::Instant::now();

    TrainingConfig {
        epoch: 30,
        train_data: mnist.trains().collect(),
        validation_data: mnist.tests().collect(),
        validation_rate: 0.1,
        batch_size: 32,
        parallel: false,
        ..Default::default()
    }
    .build()
    .fit(|batch, ctx| {
        let x = Tensor::new(
            NDArray::from_shape_vec(
                &[batch.len(), 28 * 28][..],
                batch
                    .iter()
                    .flat_map(|x| x.0)
                    .map(|x| *x as f32 / 255.0)
                    .collect(),
            )
            .unwrap(),
        );
        let t: Vec<_> = batch.iter().map(|x| x.1 as usize).collect();
        let y = mlp.call(x.clone(), true);
        let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
        if ctx.train {
            optimize(&loss, 0.001);
        }
        ctx.count(batch.len());
        ctx.add_metric(metrics::Loss::new(loss[[]], batch.len()));
        ctx.add_metric(metrics::argmax_accuracy(&t, &y));
    });

    println!("time: {:?}", start.elapsed());
}
