mod mnist;

use ndarray::prelude::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rayon::prelude::*;
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    nn::{naive_max_pooling, Conv2d, Dropout, Layer, Linear, Relu},
    *,
};

fn main() {
    let mnist = mnist::Mnist::load("./data");

    let model = BigModel::new();

    let batch_size = 10;

    let start = std::time::Instant::now();

    for epoch in 0..1000 {
        let batches: Vec<_> =
            mini_batches(&mnist.train_images, &mnist.train_labels, batch_size).collect();
        let (mut train_loss, trn_correct) = batches
            .par_iter()
            .map(|(x, t)| {
                let y = model.call(x.clone(), true);
                let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
                optimize(&loss, 0.001); // MomentumSGD: 0.1, Adam: 0.001
                (loss[[]] * t.len() as f32, count_correction(&y, &t))
            })
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        train_loss /= mnist.train_labels.len() as f32;
        let trn_acc = trn_correct as f32 / mnist.train_labels.len() as f32;

        let mut validation_loss = 0.0;
        let mut val_correct = 0;
        for (x, t) in mini_batches(&mnist.test_images, &mnist.test_labels, batch_size) {
            let y = model.call(x, false);
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            validation_loss += loss[[]] * t.len() as f32;
            val_correct += count_correction(&y, &t);
        }
        validation_loss /= mnist.test_labels.len() as f32;
        let val_acc = val_correct as f32 / mnist.test_labels.len() as f32;

        println!(
            "epoch: {}, trn_loss: {:.4}, trn_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}",
            epoch, train_loss, trn_acc, validation_loss, val_acc
        );
    }

    println!("time: {:?}", start.elapsed());
}

fn count_correction(y: &Tensor, t: &[usize]) -> usize {
    t.iter()
        .enumerate()
        .filter(|(i, t)| {
            let y = y
                .slice(s![*i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            y == **t
        })
        .count()
}

fn gen_img(img: &[u8]) -> NDArray {
    Array4::from_shape_vec(
        (img.len() / (28 * 28), 1, 28, 28),
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
                    let t =
                        Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                    AdamOptimizee::new(t)
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
                param_gen()(&[10]),
            ),
            conv2: Conv2d::new(
                [3, 3],
                [2, 2],
                [1, 1],
                param_gen()(&[10, 10, 3, 3]),
                param_gen()(&[10]),
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
}
pub struct BigModel {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub linear1: Linear,
    pub linear2: Linear,
}

impl BigModel {
    pub fn new() -> Self {
        let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
        let param_gen = {
            let rng = rng.clone();
            move || {
                let mut rng = rng.clone();
                move |shape: &[usize]| -> Param {
                    let t =
                        Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                    AdamOptimizee::new(t)
                }
            }
        };
        Self {
            conv1: Conv2d::new(
                [3, 3],
                [1, 1],
                [0, 0],
                param_gen()(&[32, 1, 3, 3]),
                param_gen()(&[32]),
            ),
            conv2: Conv2d::new(
                [3, 3],
                [1, 1],
                [0, 0],
                param_gen()(&[64, 32, 3, 3]),
                param_gen()(&[64]),
            ),
            linear1: Linear::new(64 * 12 * 12, 128, &mut param_gen(), &mut param_gen()),
            linear2: Linear::new(128, 10, &mut param_gen(), &mut param_gen()),
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
        let y = Dropout::new(0.5).call(y, train);
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
