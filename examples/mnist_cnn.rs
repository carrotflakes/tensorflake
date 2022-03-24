mod mnist;

use ndarray::prelude::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    nn::{Conv2d, Layer, Linear, Relu, naive_max_pooling},
    *,
};

fn main() {
    let mnist = mnist::Mnist::load("./data");

    let rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let param_gen = {
        let rng = rng.clone();
        move || {
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Box<dyn Fn() -> Variable> {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_tensor();
                let o = AdamOptimizee::new(t);
                Box::new(move || o.get())
            }
        }
    };

    let conv = Conv2d::new(
        [3, 3],
        // [1, 1],
        [2, 2],
        [1, 1],
        param_gen()(&[10, 1, 3, 3]),
        param_gen()(&[10]),
    );
    let conv2 = Conv2d::new(
        [3, 3],
        [2, 2],
        [1, 1],
        param_gen()(&[10, 10, 3, 3]),
        param_gen()(&[10]),
    );
    let linear = Linear::new(10 * 7 * 7, 10, &mut param_gen(), &mut param_gen());
    // let (x, t) = mini_batches(&trn_img, &trn_lbl, 10).next().unwrap();
    // let y =conv.call(vec![Variable::new(x)], true).pop().unwrap();
    // let y = call!(Relu, y);
    // let y =conv2.call(vec![y], true).pop().unwrap();
    // let y = call!(Relu, y);
    // let y = linear.call(vec![y.reshape([10, 10*7*7])], false).pop().unwrap();
    // dbg!(&*y);
    // return;

    let batch_size = 1000;

    let start = std::time::Instant::now();

    for epoch in 0..1000 {
        let mut train_loss = 0.0;
        let mut trn_correct = 0;
        for (x, t) in mini_batches(&mnist.train_images, &mnist.train_labels, batch_size) {
            let y = conv.call(vec![Variable::new(x)], true).pop().unwrap();
            // let y = naive_max_pooling(&y, [2, 2], [2, 2], [0, 0]);
            let y = call!(Relu, y);
            let y = conv2.call(vec![y], true).pop().unwrap();
            let y = call!(Relu, y);
            let y = linear
                .call(vec![y.reshape([batch_size, 10 * 7 * 7])], true)
                .pop()
                .unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            optimize(&loss, 0.001); // MomentumSGD: 0.1, Adam: 0.001
            train_loss += loss[[]];
            trn_correct += count_correction(&y, &t);
        }
        train_loss /= mnist.train_labels.len() as f32 / batch_size as f32;
        let trn_acc = trn_correct as f32 / mnist.train_labels.len() as f32;

        let mut validation_loss = 0.0;
        let mut val_correct = 0;
        for (x, t) in mini_batches(&mnist.test_images, &mnist.test_labels, batch_size) {
            let y = conv.call(vec![Variable::new(x)], false).pop().unwrap();
            // let y = naive_max_pooling(&y, [2, 2], [2, 2], [0, 0]);
            let y = call!(Relu, y);
            let y = conv2.call(vec![y], false).pop().unwrap();
            let y = call!(Relu, y);
            let y = linear
                .call(vec![y.reshape([batch_size, 10 * 7 * 7])], false)
                .pop()
                .unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            validation_loss += loss[[]];
            val_correct += count_correction(&y, &t);
        }
        validation_loss /= mnist.test_labels.len() as f32 / batch_size as f32;
        let val_acc = val_correct as f32 / mnist.test_labels.len() as f32;

        println!(
            "epoch: {}, trn_loss: {:.4}, trn_acc: {:.4}, val_loss: {:.4}, val_acc: {:.4}",
            epoch, train_loss, trn_acc, validation_loss, val_acc
        );
    }

    println!("time: {:?}", start.elapsed());
}

fn count_correction(y: &Variable, t: &[usize]) -> usize {
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

fn gen_img(img: &[u8]) -> Tensor {
    Array4::from_shape_vec(
        (img.len() / (28 * 28), 1, 28, 28),
        img.iter().map(|x| *x as f32 / 255.0).collect(),
    )
    .unwrap()
    .into_tensor()
}

fn mini_batches<'a>(
    img: &'a [u8],
    lbl: &'a [u8],
    batch_size: usize,
) -> impl Iterator<Item = (Tensor, Vec<usize>)> + 'a {
    let img = img.chunks(batch_size * 28 * 28).map(gen_img);
    let lbl = lbl
        .chunks(batch_size)
        .map(|x| x.iter().map(|x| *x as usize).collect::<Vec<_>>());
    img.zip(lbl)
}
