mod data;
mod flush_denormals;

use ndarray_rand::rand_distr::Normal;
use tensorflake::{
    initializers::Initializer,
    losses::softmax_cross_entropy,
    nn::{activations::relu, *},
    training::TrainConfig,
    *,
};

fn main() {
    // unsafe {flush_denormals::flush_denormals()};
    let mnist = data::mnist::Mnist::load("./data/mnist");

    let optimizer = optimizers::SGD::new(0.01);
    // let optimizer = optimizers::WithRegularization::new(optimizer, regularizers::L1::new(0.001));
    let init_kernel = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.1).unwrap(),
        optimizer.clone(),
    );
    let init_bias = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.0).unwrap(),
        optimizer.clone(),
    );

    let mlp = MLP::new(
        &[28 * 28, 128, 10],
        Some(Dropout::new(0.2, 42)),
        |x| relu(&x),
        init_kernel.scope("mlp_w"),
        init_bias.scope("mlp_b"),
    );

    let start = std::time::Instant::now();

    TrainConfig {
        epoch: 30,
        train_data: mnist.trains().collect(),
        validation_data: mnist.tests().collect(),
        validation_rate: 1.0,
        batch_size: 32,
        ..Default::default()
    }
    .build()
    .fit(|batch, ctx| {
        let x = Computed::new(
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
        let loss = softmax_cross_entropy(t.clone(), &y);
        ctx.finish_batch(&loss, batch.len());
        ctx.add_metric(metrics::argmax_accuracy(&t, &y));
    });

    println!("time: {:?}", start.elapsed());
}
