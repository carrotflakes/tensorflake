mod data;

use std::sync::{Arc, Mutex};

use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Normal, Uniform},
    RandomExt,
};
use tensorflake::{
    functions::*,
    losses::SoftmaxCrossEntropy,
    ndarray_util::argmax,
    nn::{activations::sigmoid, *},
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

    let mut init_kernel = initializers::InitializerWithSharedOptimizer::new(
        Normal::new(0., 0.1).unwrap(),
        optimizer.clone(),
    );
    let mut init_bias = initializers::InitializerWithSharedOptimizer::new(
        Normal::new(0., 0.0).unwrap(),
        optimizer.clone(),
    );

    let embedding_size = 64;
    let state_size = 128;
    let embedding = Embedding::new(embedding_size, vocab_size, &mut init_kernel);
    let model = Gru::new(embedding_size, state_size, &mut init_kernel);
    let linear = Linear::new(state_size, vocab_size, &mut init_kernel, &mut init_bias);
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
            let initial_state = Tensor::new(NDArray::zeros(&[strs.len(), state_size][..]));
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
            let y = model.encode(initial_state, &x);
            let yy = Concat::new(0)
                .call(y.iter().skip(eqp - 1).cloned().collect())
                .pop()
                .unwrap();
            let yy = output_fn(yy);
            let loss = call!(SoftmaxCrossEntropy::new(t), yy);
            if ctx.train {
                optimize(&loss);
            }
            ctx.count(strs.len());
            ctx.add_metric(metrics::Loss::new(loss[[]], strs.len()));
        });

        let y = model.decode(
            Tensor::new(NDArray::random_using(
                &[1, state_size][..],
                Uniform::new(0.0, 1.0),
                &mut rand_isaac::Isaac64Rng::seed_from_u64(42),
            )),
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

pub trait Cell {
    fn get_state_size(&self) -> usize;
    fn get_input_size(&self) -> usize;
    fn step(&self, x: Tensor, state: Tensor) -> (Tensor, Tensor);

    fn encode(&self, initial_state: Tensor, x: &Vec<Tensor>) -> Vec<Tensor> {
        let mut state = initial_state.clone();
        let mut outputs = vec![];
        for x in x {
            let output;
            (state, output) = self.step(x.clone(), state);
            outputs.push(output.clone());
        }
        outputs
    }

    fn decode(
        &self,
        mut state: Tensor,
        mut input: Tensor,
        output_fn: impl Fn(Tensor) -> Tensor,
        output_to_input_fn: impl Fn(Tensor) -> Tensor,
        len: usize,
    ) -> Vec<Tensor> {
        let mut outputs = vec![];
        for _ in 0..len {
            let output;
            (state, output) = self.step(input, state);
            let output = output_fn(output);
            outputs.push(output.clone());
            input = output_to_input_fn(output);
        }
        outputs
    }
}

pub struct Gru {
    pub input_size: usize,
    pub state_size: usize,
    pub ws: [Param; 3],
    pub us: [Param; 3],
    pub bs: [Param; 3],
}

impl Gru {
    pub fn new(
        input_size: usize,
        state_size: usize,
        kernel: &mut impl initializers::Initializer,
    ) -> Self {
        Self {
            input_size,
            state_size,
            ws: [
                kernel.initialize(&[input_size, state_size]),
                kernel.initialize(&[input_size, state_size]),
                kernel.initialize(&[input_size, state_size]),
            ],
            us: [
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
                kernel.initialize(&[state_size, state_size]),
            ],
            bs: [
                kernel.initialize(&[state_size]),
                kernel.initialize(&[state_size]),
                kernel.initialize(&[state_size]),
            ],
        }
    }

    pub fn all_params(&self) -> Vec<Param> {
        self.ws
            .iter()
            .chain(self.us.iter())
            .chain(self.bs.iter())
            .cloned()
            .collect()
    }
}

impl Cell for Gru {
    fn get_state_size(&self) -> usize {
        self.state_size
    }

    fn get_input_size(&self) -> usize {
        self.input_size
    }

    fn step(&self, x: Tensor, state: Tensor) -> (Tensor, Tensor) {
        let z = sigmoid(
            &(x.matmul(&self.ws[0].get_tensor())
                + state.matmul(&self.us[0].get_tensor())
                + self.bs[0].get_tensor()),
        );
        let r = sigmoid(
            &(x.matmul(&self.ws[1].get_tensor())
                + state.matmul(&self.us[1].get_tensor())
                + self.bs[1].get_tensor()),
        );
        let state = (Tensor::new(NDArray::ones(z.shape())) - z.clone()) * state.clone()
            + z * tanh(
                &(x.matmul(&self.ws[2].get_tensor())
                    + (r * state).matmul(&self.us[2].get_tensor())
                    + self.bs[2].get_tensor()),
            );
        (state.clone(), state)
    }
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
