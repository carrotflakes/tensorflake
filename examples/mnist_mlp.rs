mod mnist;

use ndarray::prelude::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    metrics::{argmax_accuracy, Metric},
    nn::*,
    *,
};

fn main() {
    let mnist = mnist::Mnist::load("./data");

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

    let batch_size = 100;

    let start = std::time::Instant::now();

    for epoch in 0..10 {
        let mut train_loss = 0.0;
        let mut trn_acc = 0.0;
        for (x, t) in mini_batches(&mnist.train_images, &mnist.train_labels, batch_size) {
            let x = Tensor::new(x);
            let y = mlp.call(x.clone(), true);
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            optimize(&loss, 0.001); // MomentumSGD: 0.1, Adam: 0.001
            train_loss += loss[[]] * t.len() as f32;
            trn_acc += argmax_accuracy(&t, &y).value() * t.len() as f32;
        }
        train_loss /= mnist.train_labels.len() as f32;
        trn_acc /= mnist.train_labels.len() as f32;

        let mut validation_loss = 0.0;
        let mut val_acc = 0.0;
        for (x, t) in mini_batches(&mnist.test_images, &mnist.test_labels, batch_size) {
            let x = Tensor::new(x);
            let y = mlp.call(x.clone(), false);
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            validation_loss += loss[[]] * t.len() as f32;
            val_acc += argmax_accuracy(&t, &y).value() * t.len() as f32;
        }
        validation_loss /= mnist.test_labels.len() as f32;
        val_acc /= mnist.test_labels.len() as f32;

        println!(
            "epoch: {}, trn_loss: {:.4}, trn_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}",
            epoch, train_loss, trn_acc, validation_loss, val_acc
        );
    }

    println!("time: {:?}", start.elapsed());
}

fn gen_img(img: &[u8]) -> NDArray {
    Array2::from_shape_vec(
        (img.len() / (28 * 28), 28 * 28),
        img.iter().map(|x| *x as f32 / 255.0).collect(),
    )
    .unwrap()
    .into_ndarray()
}

fn mini_batches<'a>(
    img: &'a [u8],
    lbl: &'a [u8],
    batch_size: usize,
) -> impl Iterator<Item = (NDArray, Vec<usize>)> + 'a {
    let img = img.chunks(batch_size * 28 * 28).map(gen_img);
    let lbl = lbl
        .chunks(batch_size)
        .map(|x| x.iter().map(|x| *x as usize).collect::<Vec<_>>());
    img.zip(lbl)
}
