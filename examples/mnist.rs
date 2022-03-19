use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array2, Array3};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ruzero::{
    functions::*,
    losses::SoftmaxCrossEntropy,
    nn::{Layer, Softmax, MLP},
    *,
};

fn main() {
    let n = 10;
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        // .training_set_length(50_000)
        .training_set_length(n as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = Array2::from_shape_vec((n, 28 * 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    // println!("{:#.1?}\n", train_data.slice(s![0, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Vec<_> = trn_lbl.iter().map(|x| *x as usize).collect();
    // println!(
    //     "The first digit is a {:?}",
    //     train_labels.slice(s![0, ..])
    // );
    dbg!(&train_labels);

    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    // let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_tensor()).named("x");

    let mlp = MLP::new(
        &[28 * 28, 100, 10],
        |xs| Sigmoid.call(xs),
        &|t: Tensor| {
            let o = MomentumSGDOptimizee::new(t, 0.9);
            Box::new(move || o.get())
        },
        &mut rng,
    );

    let x = Variable::new(train_data.into_tensor());

    let y = mlp.call(vec![x.clone()]);
    dbg!(&*y[0]);
    let y = call!(Softmax, y[0]);
    dbg!(&*y);

    for i in 0..1000 {
        let y = mlp.call(vec![x.clone()]).pop().unwrap();
        let loss = call!(SoftmaxCrossEntropy::new(train_labels.clone()), y);
        if i % 100 == 0 {
            println!("{:?}", loss[[]]);
        }
        optimize(&loss, 0.01);
    }

    let y = mlp.call(vec![x.clone()]);
    // dbg!(&*y[0]);
    let y = call!(Softmax, y[0]);
    dbg!(&*y);
}
