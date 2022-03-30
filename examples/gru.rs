mod data;

use ndarray::prelude::*;
use ndarray_rand::{
    rand::{prelude::*, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use rayon::prelude::*;
use tensorflake::{
    functions::*,
    losses::SoftmaxCrossEntropy,
    ndarray_util::{argmax, onehot},
    nn::*,
    *,
};

fn main() {
    // let mut data = data::arith::make(10000, 42, 15);
    // let vocab = plane_corpus::Vocab::new(arith::CHARS);
    let data = data::plane_corpus::load("data/corpus_en.txt").unwrap();
    let vocab = data::plane_corpus::Vocab::new(&data);
    let data = data::plane_corpus::windows(&data, 50, 25);
    let mut data = data
        .into_iter()
        .filter(|str| str.len() == 50)
        .collect::<Vec<_>>();
    let vocab_size = vocab.size();
    println!("data size: {}", data.len());

    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let mut param_gen = {
        rng.gen::<u32>();
        let mut rng = rng.clone();
        move || {
            rng.gen::<u32>();
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                Param::new(t, optimizers::SGDOptimizer::new())
            }
        }
    };

    let model = Gru::new(vocab_size, 100, &mut param_gen());
    let linear = Linear::new(100, vocab_size, &mut param_gen(), &mut param_gen());

    let start = std::time::Instant::now();

    // let mut gradients = GradientsAccumulator::new();
    for mut ctx in ExecutionContextIter::new(100, Some(data.len())) {
        if !ctx.train {
            // TODO
            continue;
        }
        data.shuffle(&mut rng);
        let metrics = data
            .par_chunks(20)
            .map(|strs| {
                let initial_state = Tensor::new(NDArray::zeros(&[strs.len(), 100][..]));
                let eqp = 10;
                let mut x = vec![vec![]; 50 - 1];
                let mut t = vec![vec![]; 50 - eqp];
                for str in strs.iter() {
                    let y = vocab.encode(str);
                    for (j, c) in y.iter().take(str.len() - 1).enumerate() {
                        x[j].push(*c);
                    }
                    for (j, c) in y.iter().skip(eqp + 1).enumerate() {
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
                        onehot(&Array::from_shape_vec([strs.len()], x).unwrap(), vocab_size).into()
                    })
                    .collect::<Vec<_>>();
                let t = t.into_iter().flatten().collect();
                let y = model.encode(initial_state, &x);
                let yy = Concat::new(0)
                    .call(y.iter().skip(eqp).cloned().collect())
                    .pop()
                    .unwrap();
                let yy = linear.call(yy, true);
                let loss = call!(SoftmaxCrossEntropy::new(t), yy);
                optimize(&loss, 0.1 * 0.95f32.powi(ctx.epoch as i32));
                let mut metrics = Metrics::new();
                metrics.count(strs.len());
                metrics.add(metrics::Loss::new(loss[[]], strs.len()));
                metrics
            })
            .reduce(
                || Metrics::new(),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            );
        ctx.merge_metrics(metrics);
        let y = model.decode(
            Tensor::new(NDArray::random_using(
                &[1, 100][..],
                Uniform::new(0.0, 1.0),
                &mut rng,
            )),
            |x| linear.call(x, false),
            |x| onehot(&argmax(&*x), vocab_size).into(),
            50,
        );
        let str: String = y
            .iter()
            .map(|x| vocab.decode(&argmax(&*x).into_raw_vec()))
            .collect();
        println!("{}", str);
        ctx.print_result();
    }

    println!("time: {:?}", start.elapsed());
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
        param_gen: &mut impl FnMut(&[usize]) -> Param,
    ) -> Self {
        Self {
            input_size,
            state_size,
            ws: [
                param_gen(&[input_size, state_size]),
                param_gen(&[input_size, state_size]),
                param_gen(&[input_size, state_size]),
            ],
            us: [
                param_gen(&[state_size, state_size]),
                param_gen(&[state_size, state_size]),
                param_gen(&[state_size, state_size]),
            ],
            bs: [
                param_gen(&[state_size]),
                param_gen(&[state_size]),
                param_gen(&[state_size]),
            ],
        }
    }

    pub fn step(&self, x: Tensor, state: Tensor) -> Tensor {
        let z = call!(
            Sigmoid,
            x.matmul(&self.ws[0].get_tensor())
                + state.matmul(&self.us[0].get_tensor())
                + self.bs[0].get_tensor()
        );
        let r = call!(
            Sigmoid,
            x.matmul(&self.ws[1].get_tensor())
                + state.matmul(&self.us[1].get_tensor())
                + self.bs[1].get_tensor()
        );
        (Tensor::new(NDArray::ones(z.shape())) - z.clone()) * state.clone()
            + z * call!(
                Tanh,
                x.matmul(&self.ws[2].get_tensor())
                    + (r * state).matmul(&self.us[2].get_tensor())
                    + self.bs[2].get_tensor()
            )
    }

    pub fn encode(&self, initial_state: Tensor, x: &Vec<Tensor>) -> Vec<Tensor> {
        let batch_size = x[0].shape()[0];
        assert_eq!(initial_state.shape(), &[batch_size, self.state_size]);
        for x in x {
            assert_eq!(x.shape(), &[batch_size, self.input_size]);
        }
        let mut state = initial_state.clone();
        let mut outputs = vec![];
        for x in x {
            state = self.step(x.clone(), state);
            outputs.push(state.clone());
        }
        outputs
    }

    pub fn decode(
        &self,
        mut state: Tensor,
        output_fn: impl Fn(Tensor) -> Tensor,
        output_to_input_fn: impl Fn(Tensor) -> Tensor,
        len: usize,
    ) -> Vec<Tensor> {
        let mut outputs = vec![];
        for _ in 0..len {
            let output = output_fn(state.clone());
            outputs.push(output.clone());
            let input = output_to_input_fn(output);
            state = self.step(input, state);
        }
        outputs
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
