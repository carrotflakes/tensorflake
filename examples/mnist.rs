use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array2};
use ndarray_rand::rand::SeedableRng;
use tensorflake::{
    losses::SoftmaxCrossEntropy,
    nn::{Layer, Relu, MLP},
    *,
};

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(10_000)
        .test_set_length(0)
        .finalize();

    let gen_img = |img: &[u8]| {
        Array2::from_shape_vec(
            (img.len() / (28 * 28), 28 * 28),
            img.iter().map(|x| *x as f32 / 255.0).collect(),
        )
        .unwrap()
    };

    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let mlp = MLP::new(
        &[28 * 28, 128, 10],
        Some(0.2),
        |xs| Relu.call(xs),
        &|t: Tensor| {
            // let o = MomentumSGDOptimizee::new(t, 0.9);
            let o = AdamOptimizee::new(t);
            Box::new(move || o.get())
        },
        &mut rng,
    );

    let batch_size = 1000;

    let start = std::time::Instant::now();

    for epoch in 0..1000 {
        let mut train_loss = 0.0;
        let mut trn_correct = 0;
        for (x, t) in {
            let img = trn_img.chunks(batch_size * 28 * 28).map(gen_img);
            let lbl = trn_lbl
                .chunks(batch_size)
                .map(|x| x.iter().map(|x| *x as usize).collect::<Vec<_>>());
            img.zip(lbl)
        } {
            let x = Variable::new(x.into_tensor());
            let y = mlp.call(vec![x.clone()], true).pop().unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            optimize(&loss, 0.001); // MomentumSGD: 0.1, Adam: 0.001
            train_loss += loss[[]];
            trn_correct += count_correction(&y, &t);
        }
        train_loss /= trn_lbl.len() as f32 / batch_size as f32;
        let trn_acc = trn_correct as f32 / trn_lbl.len() as f32;

        let mut validation_loss = 0.0;
        let mut val_correct = 0;
        for (x, t) in {
            let img = val_img.chunks(batch_size * 28 * 28).map(gen_img);
            let lbl = val_lbl
                .chunks(batch_size)
                .map(|x| x.iter().map(|x| *x as usize).collect::<Vec<_>>());
            img.zip(lbl)
        } {
            let x = Variable::new(x.into_tensor());
            let y = mlp.call(vec![x.clone()], false).pop().unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            validation_loss += loss[[]];
            val_correct += count_correction(&y, &t);
        }
        validation_loss /= val_lbl.len() as f32 / batch_size as f32;
        let val_acc = val_correct as f32 / val_lbl.len() as f32;

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
