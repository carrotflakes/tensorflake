use ndarray::array;
use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Normal,
};
use tensorflake::{
    initializers::Initializer,
    losses::naive_mean_squared_error,
    nn::{activations::sigmoid, *},
    training::TrainConfig,
    *,
};

fn main() {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    let x = (0..10000)
        .map(|_| rng.gen_range(0.0f32..1.0))
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|x| (x * 2.0 * std::f32::consts::PI).sin() + rng.gen_range(-0.5..0.5))
        .collect::<Vec<_>>();

    let optimizer = optimizers::SGD::new(0.1);
    let init_kernel = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.1).unwrap(),
        optimizer.clone(),
    );
    let init_bias = initializers::InitializerWithOptimizer::new(
        Normal::new(0.0, 0.0).unwrap(),
        optimizer.clone(),
    );

    let l1 = Linear::new(1, 10, init_kernel.scope("l1_w"), Some(init_bias.scope("l1_b")));
    let l2 = Linear::new(10, 1, init_kernel.scope("l2_w"), Some(init_bias.scope("l2_b")));

    let start = std::time::Instant::now();

    TrainConfig {
        epoch: 100,
        train_data: x.into_iter().zip(y.into_iter()).collect(),
        batch_size: 100,
        shuffle: true,
        ..Default::default()
    }
    .build()
    .fit(|batch, ctx| {
        let x = ComputedNDA::new(
            NDArray::from_shape_vec(
                &[batch.len(), 1][..],
                batch.iter().map(|x| x.0).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let t = ComputedNDA::new(
            NDArray::from_shape_vec(
                &[batch.len(), 1][..],
                batch.iter().map(|x| x.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let h = l1.call(x, true);
        let h = sigmoid(&h);
        let y = l2.call(h, true);

        let loss = naive_mean_squared_error(y.clone(), t.clone());
        ctx.finish_batch(&loss, batch.len());
    });

    for i in 0..20 {
        let x = ComputedNDA::new(array![[i as f32 / 20.0]].into_ndarray());
        let h = l1.call(x.clone(), false);
        let h = sigmoid(&h);
        let y = l2.call(h, false);
        println!("{}", y[[0, 0]]);
    }

    println!("elapsed: {:?}", start.elapsed());
}
