use ndarray::Array;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use tensorflake::{functions::*, losses::naive_mean_squared_error, *};

fn main() {
    let n = 10;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    // make dataset
    let x =
        ComputedNDA::new(Array::random_using((n, 1, 1), Uniform::new(0., 1.), &mut rng).into_ndarray());
    let y = (&x * &ComputedNDA::new(scalar(2.0))) + ComputedNDA::new(Array::zeros((n, 1, 1)).into_ndarray());

    // dbg!(&*x);
    // dbg!(&*y);

    let mut w = backprop(ndarray::array![[[0.0]]].into_ndarray()).named("w");
    let mut b = backprop(ndarray::array![0.0].into_ndarray()).named("b");

    let predict = |w: ComputedNDA, b: ComputedNDA, x: ComputedNDA| matmul_add(&x, &w.broadcast(vec![n, 1, 1]), &b);

    for i in 0..100 {
        let y_ = predict(w.clone(), b.clone(), x.clone());
        // dbg!(&*y_);

        let loss = naive_mean_squared_error(y.clone(), y_.clone());
        println!("loss: {}", loss[[]]);

        if i == 0 {
            graph(&[loss.clone()]);
        }

        let gs = gradients(&[loss.clone()], &[w.clone(), b.clone()], false);

        let lr = 0.01;
        w = backprop((&*w - &*gs[0] * lr).into_ndarray());
        b = backprop((&*b - &*gs[1] * lr).into_ndarray());
    }
}

fn graph(vars: &[ComputedNDA]) {
    let f = std::fs::File::create("graph.dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    tensorflake::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
