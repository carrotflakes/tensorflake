mod data;

use std::sync::{Arc, Mutex};

use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Normal, Uniform},
    RandomExt,
};
use tensorflake::{
    functions::*,
    initializers::Initializer,
    losses::softmax_cross_entropy,
    ndarray_util::argmax,
    nn::{
        rnn::{Cell, Gru},
        *,
    },
    training::TrainConfig,
    *,
};

fn main() {
    // let mut data = data::arith::make(10000, 42, 15);
    // let vocab = plane_corpus::Vocab::new(arith::CHARS);
    let data = data::plane_corpus::load("data/corpus_en.txt").unwrap();
    let vocab = data::plane_corpus::Vocab::new(&data);
    let data = data::plane_corpus::windows(&data, 50, 25);
    let data = data
        .into_iter()
        .filter(|str| str.len() == 50)
        .collect::<Vec<_>>();
    let vocab_size = vocab.size();
    println!("data size: {}", data.len());
    println!("vocab size: {}", vocab_size);

    // let optimizer = optimizers::SGDOptimizer::new();
    // let lr = 0.1;
    let optimizer = Arc::new(Mutex::new(optimizers::AdamOptimizer::new()));
    // let optimizer = optimizers::WithRegularization::new(optimizer, regularizers::L2::new(0.001));
    let lr = 0.0001;

    let norm =
        normalization::Normalization::new(vec![0, 1], 0.001, optimizers::AdamOptimizer::new());

    let init_kernel = initializers::InitializerWithSharedOptimizer::new(
        Normal::new(0., 0.1).unwrap(),
        optimizer.clone(),
    );
    let init_bias = initializers::InitializerWithSharedOptimizer::new(
        Normal::new(0., 0.0).unwrap(),
        optimizer.clone(),
    );

    let embedding_size = 64;
    let state_size = 128;
    let embedding = Embedding::new(embedding_size, vocab_size, init_kernel.scope("embedding"));
    let model = Gru::new(embedding_size, state_size, init_kernel.scope("gru"));
    let linear = Linear::new(
        state_size,
        vocab_size,
        init_kernel.scope("out_proj_w"),
        Some(init_bias.scope("out_proj_b")),
    );
    // let output_fn = |x: Tensor| linear.call(x, true);
    let output_fn = |x: Tensor| linear.call(norm.call(x, true), true);

    let start = std::time::Instant::now();

    let mut train = TrainConfig {
        epoch: 30,
        train_data: data,
        batch_size: 100,
        parallel: true,
        ..Default::default()
    }
    .build();
    while !train.is_end() {
        optimizer.lock().unwrap().learning_rate = lr * 0.95f32.powi(train.epoch as i32);
        train.fit_one_epoch(|strs, ctx| {
            let initial_state = model.initial_state(strs.len());
            let eqp = 10;
            let mut x = vec![vec![]; 50 - 1];
            let mut t = vec![vec![]; 50 - eqp];
            for str in strs.iter() {
                let y = vocab.encode(str);
                for (j, c) in y.iter().take(str.len() - 1).enumerate() {
                    x[j].push(*c);
                }
                for (j, c) in y.iter().skip(eqp).enumerate() {
                    t[j].push(*c);
                }
            }
            if x[0].len() != x[50 - 2].len() {
                for s in strs {
                    println!("{}", s.len());
                }
                panic!("length unmatch")
            }
            let x = x
                .into_iter()
                .map(|x| {
                    // onehot(&Array::from_shape_vec([strs.len()], x).unwrap(), vocab_size).into()
                    embedding.call(x, ctx.train)
                })
                .collect::<Vec<_>>();
            let t = t.into_iter().flatten().collect();
            let y = model.encode(initial_state, &x).1;
            let yy = Concat::new(0)
                .call(y.iter().skip(eqp - 1).cloned().collect())
                .pop()
                .unwrap();
            let yy = output_fn(yy);
            let loss = softmax_cross_entropy(t, &yy);
            if ctx.train {
                optimize(&loss);
            }
            ctx.count(strs.len());
            ctx.add_metric(metrics::Loss::new(loss[[]], strs.len()));
        });

        let y = model.decode(
            model.initial_state(1),
            Tensor::new(NDArray::random_using(
                &[1, embedding_size][..],
                Uniform::new(0.0, 1.0),
                &mut rand_isaac::Isaac64Rng::seed_from_u64(42),
            )),
            output_fn,
            |x|
            //onehot(&argmax(&*x), vocab_size).into()
            embedding.call(argmax(&*x).into_raw_vec(), false),
            50,
        );
        let str: String = y
            .iter()
            .map(|x| vocab.decode(&argmax(&*x).into_raw_vec()))
            .collect();
        println!("{}", str);
    }

    println!("time: {:?}", start.elapsed());
}

#[allow(dead_code)]
fn graph(vars: &[Tensor], name: impl ToString) {
    let f = std::fs::File::create(name.to_string() + ".dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    tensorflake::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
