// Forward-forward algorithm
//https://www.cs.toronto.edu/~hinton/FFA13.pdf
//
// for i in {0..5}; do convert ffa_$i.pnm ffa_$i.png; done

mod data;
mod flush_denormals;

use ndarray_rand::rand_distr::Normal;
use tensorflake::{
    initializers::{
        random_initializer::RandomInitializer, with_optimizer::InitializerWithOptimizer, Scope,
    },
    ndarray_util::onehot,
    nn::{
        activations::{relu, sigmoid},
        *,
    },
    *,
};

fn main() {
    // unsafe {flush_denormals::flush_denormals()};
    let mnist = data::mnist::Mnist::load("./data/mnist");

    let optimizer = optimizers::Adam::new();
    // let optimizer = optimizers::WithRegularization::new(optimizer, regularizers::L1::new(0.001));
    let init_kernel = InitializerWithOptimizer::new(
        RandomInitializer::new(Normal::new(0.0, 0.01).unwrap()),
        optimizer.clone(),
    );
    let init_bias = InitializerWithOptimizer::new(
        RandomInitializer::new(Normal::new(0.01, 0.0).unwrap()),
        optimizer.clone(),
    );

    let linear = Linear::new(
        10 + 28 * 28,
        32,
        init_kernel.scope("layer_w"),
        Some(init_bias.scope("layer_b")),
    );

    let mut count = 0;
    let mut total_loss = 0.0;

    test(&linear, &mnist);

    for (pos, neg) in mnist.trains().zip(mnist.trains().skip(100)) {
        let goodness = forward(&linear, pos.1 as usize, &pos.0);
        let neg_goodness = forward(&linear, (neg.1 as usize + 1 + count % 9) % 10, neg.0);

        let loss = &neg_goodness - &goodness;
        optimize(&loss);
        total_loss += loss[[]];
        if count % 1000 == 0 {
            println!(
                "pos: {}, neg: {}, loss: {}",
                goodness[[]],
                neg_goodness[[]],
                total_loss
            );
            total_loss = 0.0;
        }
        if (count + 1) % 10000 == 0 {
            output_image(&*linear.w.get(), count / 10000);
            test(&linear, &mnist);
        }
        count += 1;
    }
}

fn forward(linear: &Linear, label: usize, image: &[u8]) -> Computed<NDArray> {
    let lbl = onehot(&ndarray::arr1(&vec![label]), 10);
    let x = NDArray::from_shape_vec(
        &[1, 10 + 28 * 28][..],
        lbl.iter()
            .cloned()
            .chain(image.iter().map(|x| *x as f32 / 255.0))
            .collect(),
    )
    .unwrap();
    let y = linear.call(x.into(), true);
    let y = relu(&y);
    let goodness = sigmoid(&(y.pow_const(2.0).sum(vec![0, 1], false) - scalar(5.0).into()));
    goodness
}

fn test(linear: &Linear, mnist: &data::mnist::Mnist) {
    let mut correct_count = 0;
    let mut count = 0;
    for (img, lbl) in mnist.tests() {
        let (l, _) = (0..10)
            .map(|label| {
                let goodness = forward(linear, label, img);
                (label, goodness[[]])
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        count += 1;
        if l == lbl as usize {
            correct_count += 1;
        }
    }
    println!(
        "accuracy: {} / {} = {}",
        correct_count,
        count,
        correct_count as f64 / count as f64
    );
}

fn output_image(w: &NDArray, i: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(format!("ffa_{}.pnm", i)).unwrap();
    let num = 32;
    writeln!(&mut f, "P2\n{} {}\n15", 28, (1 + 28) * num).unwrap();
    for i in 0..num {
        // print label
        let sliced = w.slice(ndarray::s![..10, i]);
        let min = sliced.iter().fold(0.0 / 0.0, |m, v| v.min(m));
        let max = sliced.iter().fold(0.0 / 0.0, |m, v| v.max(m));
        for x in sliced.iter().chain(vec![min; 28 - 10].iter()) {
            write!(&mut f, "{} ", ((x - min) / (max - min) * 16.0) as usize).unwrap();
        }

        // print image
        let sliced = w.slice(ndarray::s![10.., i]);
        let min = sliced.iter().fold(0.0 / 0.0, |m, v| v.min(m));
        let max = sliced.iter().fold(0.0 / 0.0, |m, v| v.max(m));
        for (i, x) in sliced.iter().enumerate() {
            if i % 28 == 0 {
                writeln!(&mut f).unwrap();
            }
            write!(&mut f, "{} ", ((x - min) / (max - min) * 16.0) as usize).unwrap();
        }
    }
    writeln!(&mut f).unwrap();
}
