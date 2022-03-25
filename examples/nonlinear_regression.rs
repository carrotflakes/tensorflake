use ndarray::{array, Array};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{functions::*, nn::*, *};

fn main() {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let n = 100;

    let x = Tensor::new(Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray())
        .named("x");
    let y = call!(
        Add,
        call!(Sin, call!(Mul, x, Tensor::new(scalar(2.0 * 3.14)))),
        Tensor::new(Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray())
    )
    .named("y");

    let l1 = Layer::new(1, 10);
    let l2 = Layer::new(10, 1);

    let trainables = l1
        .all_params()
        .into_iter()
        .chain(l2.all_params())
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    // let mut ll = 1000.0;

    for i in 0..10000 {
        let h = l1.forward(x.clone());
        let h = call!(Sigmoid, h).named("hidden");
        let y_ = l2.forward(h);
        // dbg!(&*y_);
        if i == 0 {
            graph(&[y_.clone()], "graph");
        }

        let loss = mean_squared_error(y.clone(), y_.clone());
        if i % 1000 == 0 {
            println!("loss: {}", loss[[]]);
        }
        // if loss[[]] > ll {
        //     dbg!("loss is not decreasing");
        //     return;
        // }
        // ll = loss[[]];

        // graph(&[loss.clone()], format!("loss{}", i));

        let gs = gradients(&[loss.clone()], &trainables, false);
        let lr = Tensor::new(scalar(0.1));
        for (t, g) in trainables.iter().zip(gs.into_iter()) {
            unsafe { t.add_assign(&call!(Mul, g, call!(Neg, lr))) };
        }
    }
    for i in 0..20 {
        let x = Tensor::new(array![[i as f32 / 20.0]].into_ndarray());
        let h = l1.forward(x);
        let h = naive_sigmoid(h).named("hidden");
        let y_ = l2.forward(h);
        println!("{}", &*y_);
    }
    println!("elapsed: {:?}", start.elapsed());
}

fn mean_squared_error(x0: Tensor, x1: Tensor) -> Tensor {
    let x = call!(Pow::new(2.0), call!(Sub, x0, x1));
    call!(
        Div,
        call!(Sum::new((0..x.ndim()).collect(), false), x),
        Tensor::new(scalar(x.shape().iter().product::<usize>() as f32))
    )
}

pub struct Layer {
    pub w: Tensor,
    pub b: Tensor,
}

impl Layer {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            w: backprop(Array::random((input, output), Uniform::new(0., 0.01)).into_ndarray())
                .named("param w"),
            b: backprop(Array::zeros(output).into_ndarray()).named("param b"),
        }
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        call!(Add, call!(Matmul, x, self.w), self.b).named("return")
    }

    pub fn all_params(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }
}

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
