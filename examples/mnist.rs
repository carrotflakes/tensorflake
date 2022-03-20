use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array2, Array3, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ruzero::{
    functions::*,
    losses::SoftmaxCrossEntropy,
    nn::{Layer, Softmax, MLP},
    *,
};

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let mlp = MLP::new(
        &[28 * 28, 100, 10],
        |xs| Sigmoid.call(xs),
        &|t: Tensor| {
            let o = MomentumSGDOptimizee::new(t, 0.9);
            Box::new(move || o.get())
        },
        &mut rng,
    );

    let batch_size = 1000;

    let start = std::time::Instant::now();

    for epoch in 0..20 {
        let mut train_loss = 0.0;
        for i in 0..trn_lbl.len() / batch_size {
            let x = Array2::from_shape_vec(
                (batch_size, 28 * 28),
                trn_img
                    .iter()
                    .skip(i * 28 * 28)
                    .take(batch_size * 28 * 28)
                    .cloned()
                    .collect(),
            )
            .unwrap()
            .map(|x| *x as f32 / 256.0);
            let t = trn_lbl
                .iter()
                .skip(i)
                .take(batch_size)
                .map(|x| *x as usize)
                .collect();
            let x = Variable::new(x.into_tensor());
            let y = mlp.call(vec![x.clone()]).pop().unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t), y);
            optimize(&loss, 0.1);
            train_loss += loss[[]];
        }
        train_loss /= trn_lbl.len() as f32 / batch_size as f32;

        let mut validation_loss = 0.0;
        let mut correct_num = 0;
        for i in 0..val_lbl.len() / batch_size {
            let x = Array2::from_shape_vec(
                (batch_size, 28 * 28),
                val_img
                    .iter()
                    .skip(i * 28 * 28)
                    .take(batch_size * 28 * 28)
                    .cloned()
                    .collect(),
            )
            .unwrap()
            .map(|x| *x as f32 / 256.0);
            let t: Vec<_> = val_lbl
                .iter()
                .skip(i)
                .take(batch_size)
                .map(|x| *x as usize)
                .collect();
            let x = Variable::new(x.into_tensor());
            let y = mlp.call(vec![x.clone()]).pop().unwrap();
            let loss = call!(SoftmaxCrossEntropy::new(t.clone()), y);
            validation_loss += loss[[]];
            for (i, t) in t.iter().cloned().enumerate() {
                let y = y
                    .slice(s![i, ..])
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                if y == t {
                    correct_num += 1;
                }
            }
        }
        validation_loss /= val_lbl.len() as f32 / batch_size as f32;
        let accuracy = correct_num as f32 / val_lbl.len() as f32;

        println!(
            "epoch: {}, train_loss: {:.4}, validation_loss: {:.4}, accuracy: {:.4}",
            epoch, train_loss, validation_loss, accuracy
        );
    }

    println!("time: {:?}", start.elapsed());
}
