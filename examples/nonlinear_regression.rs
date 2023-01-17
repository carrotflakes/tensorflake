use ndarray::{array, Array};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{
    initializers::{
        random_initializer::RandomInitializer, with_optimizer::InitializerWithOptimizer, Scope,
    },
    losses::naive_mean_squared_error,
    nn::{
        activations::{naive_sigmoid, sigmoid},
        *,
    },
    *,
};

fn main() {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let n = 100;

    let x = ComputedNDA::new(
        Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray(),
    )
    .named("x");
    let y = ((&x * &ComputedNDA::new(scalar(2.0 * 3.14))).sin()
        + ComputedNDA::new(
            Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray(),
        ))
    .named("y");

    let init = InitializerWithOptimizer::new(
        RandomInitializer::new(Uniform::new(0., 0.01)),
        optimizers::Adam::new(),
    );

    let l1 = Linear::new(1, 10, init.scope("l1_w"), Some(init.scope("l1_b")));
    let l2 = Linear::new(10, 1, init.scope("l2_w"), Some(init.scope("l2_b")));

    let start = std::time::Instant::now();

    // let mut ll = 1000.0;

    for i in 0..10000 {
        let h = l1.call(x.clone(), true);
        let h = sigmoid(&h).named("hidden");
        let y_ = l2.call(h, true);
        // dbg!(&*y_);
        if i == 0 {
            graph(&[y_.clone()], "graph");
        }

        let loss = naive_mean_squared_error(y.clone(), y_.clone());
        if i % 1000 == 0 {
            println!("loss: {}", loss[[]]);
        }
        // if loss[[]] > ll {
        //     dbg!("loss is not decreasing");
        //     return;
        // }
        // ll = loss[[]];

        // graph(&[loss.clone()], format!("loss{}", i));

        optimize(&loss);
    }
    for i in 0..20 {
        let x = ComputedNDA::new(array![[i as f32 / 20.0]].into_ndarray());
        let h = l1.call(x.clone(), false);
        let h = naive_sigmoid(h).named("hidden");
        let y_ = l2.call(h, false);
        println!("{}", &*y_);
    }
    println!("elapsed: {:?}", start.elapsed());
}

fn graph(vars: &[ComputedNDA], name: impl ToString) {
    let f = std::fs::File::create(name.to_string() + ".dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    tensorflake::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
