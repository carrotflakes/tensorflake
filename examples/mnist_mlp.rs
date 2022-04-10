mod data;

use ndarray_rand::rand_distr::Normal;
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    nn::{activations::Relu, *},
    training::TrainConfig,
    *,
};

fn main() {
    let mnist = data::mnist::Mnist::load("./data/mnist");

    let optimizer = optimizers::AdamOptimizer::new();
    let mut init_kernel = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.1).unwrap(),
        optimizer.clone(),
    );
    let mut init_bias = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.0).unwrap(),
        optimizer.clone(),
    );

    let mlp = MLP::new(
        &[28 * 28, 128, 10],
        Some(Dropout::new(0.2, 42)),
        |x| Relu.call(vec![x]).pop().unwrap(),
        &mut init_kernel,
        &mut init_bias,
    );

    let start = std::time::Instant::now();

    TrainConfig {
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
        ctx.finish_batch(&loss, batch.len());
        ctx.add_metric(metrics::argmax_accuracy(&t, &y));
    });

    println!("time: {:?}", start.elapsed());
}
