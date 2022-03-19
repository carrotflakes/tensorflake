use ndarray::Array;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ruzero::{functions::*, *};

fn main() {
    let n = 10;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    // make dataset
    let x =
        Variable::new(Array::random_using((n, 1, 1), Uniform::new(0., 1.), &mut rng).into_tensor());
    let y = call!(
        Add,
        call!(Mul, x, Variable::new(scalar(2.0))),
        Variable::new(Array::zeros((n, 1, 1)).into_tensor())
    );

    // dbg!(&*x);
    // dbg!(&*y);

    let mut w = backprop(ndarray::array![[[0.0]]].into_tensor()).named("w");
    let mut b = backprop(ndarray::array![0.0].into_tensor()).named("b");

    let predict = |w: Variable, b: Variable, x: Variable| {
        call!(
            Add,
            call!(Matmul, x, call!(BroadcastTo::new(vec![n, 1, 1]), w)),
            b
        )
    };

    for i in 0..100 {
        let y_ = predict(w.clone(), b.clone(), x.clone());
        // dbg!(&*y_);

        let loss = mean_squared_error(y.clone(), y_.clone());
        println!("loss: {}", loss[[]]);

        if i == 0 {
            graph(&[loss.clone()]);
        }

        let gs = gradients(&[loss.clone()], &[w.clone(), b.clone()], false);

        let lr = 0.01;
        w = backprop((&*w - &*gs[0] * lr).into_tensor());
        b = backprop((&*b - &*gs[1] * lr).into_tensor());
    }
}

fn mean_squared_error(x0: Variable, x1: Variable) -> Variable {
    let x = call!(Pow::new(2.0), call!(Sub, x0, x1));
    call!(
        Div,
        call!(SumTo::new((0..x.ndim()).collect()), x),
        Variable::new(scalar(x.shape().iter().product::<usize>() as f32))
    )
}

fn graph(vars: &[Variable]) {
    let f = std::fs::File::create("graph.dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    ruzero::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
