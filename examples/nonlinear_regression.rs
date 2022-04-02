use ndarray::{array, Array};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{
    functions::*,
    nn::{
        activations::{naive_sigmoid, Sigmoid},
        *,
    },
    *,
};

fn main() {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let n = 100;

    let x =
        Tensor::new(Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray())
            .named("x");
    let y = call!(
        Add,
        call!(Sin, call!(Mul, x, Tensor::new(scalar(2.0 * 3.14)))),
        Tensor::new(Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_ndarray())
    )
    .named("y");

    let param_gen = {
        let rng = rng.clone();
        move || {
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                Param::new(t, optimizers::AdamOptimizer::new())
            }
        }
    };

    let l1 = Linear::new(1, 10, &mut param_gen(), &mut param_gen());
    let l2 = Linear::new(10, 1, &mut param_gen(), &mut param_gen());

    let start = std::time::Instant::now();

    // let mut ll = 1000.0;

    for i in 0..10000 {
        let h = l1.call(x.clone(), true);
        let h = call!(Sigmoid, h).named("hidden");
        let y_ = l2.call(h, true);
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

        optimize(&loss, 0.1);
    }
    for i in 0..20 {
        let x = Tensor::new(array![[i as f32 / 20.0]].into_ndarray());
        let h = l1.call(x.clone(), false);
        let h = naive_sigmoid(h).named("hidden");
        let y_ = l2.call(h, false);
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
