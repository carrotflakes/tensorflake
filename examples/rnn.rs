mod data;

use data::arith;
use ndarray_rand::{
    rand::{prelude::SliceRandom, SeedableRng},
    rand_distr::Uniform,
};
use tensorflake::{
    functions::*,
    initializers::{Initializer, InitializerWithOptimizer},
    losses::softmax_cross_entropy,
    ndarray_util::{argmax, onehot},
    nn::*,
    *,
};

fn main() {
    let mut data = arith::make(10000, 42, 15);
    // let data = (0..10000)
    //     .map(|i| format!("=({}{}{})", i % 6 + 1, i % 6 + 1, i % 6 + 1))
    //     .collect::<Vec<_>>();
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    let model = Model::new(arith::VOCAB_SIZE);

    let start = std::time::Instant::now();

    {
        let str = "1+2=3";
        let y = model.call(arith::encode(&str[..str.len() - 1]), true);
        let y = concat(&y, 0);
        let loss = softmax_cross_entropy(arith::encode(&str[1..]), &y);
        graph(&[loss], "rnn");
    }

    let mut gradients = GradientsAccumulator::new();
    for e in 0..100 {
        data.shuffle(&mut rng);
        for i in 0..data.len() {
            let str = &data[i];
            let eqp = str.chars().position(|c| c == '=').unwrap();
            let y = model.call(arith::encode(&str[..str.len() - 1]), true);
            let yy = concat(&y.iter().skip(eqp).cloned().collect::<Vec<_>>(), 0);
            let loss = softmax_cross_entropy(arith::encode(&str[eqp + 1..]), &yy);
            gradients.compute(&loss);
            if i % 10 == 0 {
                gradients.optimize();
            }
            if i % 5000 == 0 {
                // println!("{:?}", &*y[1]);
                let y = concat(&y, 0);
                let v = argmax(&*y).into_raw_vec();
                // println!("{:?}", v);
                println!("{}", str);
                println!(" {}", arith::decode(&v));
                dbg!(loss[[]]);
            }
        }
        // for (p, t) in gradients.table.iter() {
        //     dbg!(&**t);
        // }
        println!("epoch: {}", e);
    }

    println!("time: {:?}", start.elapsed());
}

pub struct Model {
    pub vocab_size: usize,
    pub initial: Param,
    pub enb: Linear,
    pub linear: Linear,
    pub output: Linear,
}

impl Model {
    pub fn new(vocab_size: usize) -> Self {
        let init =
            InitializerWithOptimizer::new(Uniform::new(0., 0.01), optimizers::SGD::new(0.01));
        let state_size = 200;
        Self {
            vocab_size,
            initial: init.scope("initial").initialize(&[1, state_size]),
            enb: Linear::new(
                vocab_size,
                state_size,
                init.scope("enb"),
                Some(init.scope("enb")),
            ),
            linear: Linear::new(
                state_size,
                state_size,
                init.scope("linear"),
                Some(init.scope("linear")),
            ),
            output: Linear::new(
                state_size,
                vocab_size,
                init.scope("output"),
                Some(init.scope("output")),
            ),
        }
    }

    pub fn call(&self, x: Vec<usize>, train: bool) -> Vec<Computed> {
        let mut state = self.initial.get();
        let mut outputs = vec![];
        for x in x {
            let enb = self
                .enb
                .call(onehot(&ndarray::arr1(&[x]), self.vocab_size).into(), train);
            let concated = &enb + &state;
            state = self.linear.call(concated, train);
            state = state.tanh();
            outputs.push(self.output.call(state.clone(), train).named("output"));
        }
        outputs
    }

    pub fn all_params(&self) -> Vec<Param> {
        self.enb
            .all_params()
            .into_iter()
            .chain(self.linear.all_params())
            .collect()
    }
}

fn graph(vars: &[Computed], name: impl ToString) {
    let f = std::fs::File::create(name.to_string() + ".dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    tensorflake::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
