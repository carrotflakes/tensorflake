use ndarray::prelude::*;
use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use rayon::prelude::*;
use tensorflake::{
    functions::*,
    losses::naive_mean_squared_error,
    nn::{Layer, MLP},
    *,
};

fn main() {
    let n = 100000;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    let param_gen = {
        let rng = rng.clone();
        move || {
            let mut rng = rng.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                Param::new(t, optimizers::SGDOptimizer::new())
            }
        }
    };

    let x = (0..n)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect::<Vec<f32>>();
    let y = x
        .into_iter()
        .map(|x| (x, (x * std::f32::consts::PI * 2.0).sin()))
        .collect::<Vec<_>>();

    let mlp = MLP::new(
        &[1, 20, 20, 1],
        None,
        |x| Tanh.call(vec![x]).pop().unwrap(),
        &mut param_gen(),
        &mut param_gen(),
    );

    for e in 0..30 {
        let loss = y
            .par_chunks(100)
            .map(|data| {
                let xs = Array2::from_shape_vec(
                    [100, 1],
                    data.iter().map(|&(x, _)| x).collect::<Vec<_>>(),
                )
                .unwrap()
                .into_ndarray();
                let ts = Array2::from_shape_vec(
                    [100, 1],
                    data.iter().map(|&(_, y)| y).collect::<Vec<_>>(),
                )
                .unwrap()
                .into_ndarray();
                let ys = mlp.call(xs.into(), true);
                let loss = naive_mean_squared_error(ys.into(), ts.into());
                optimize(&loss, 0.01 * 0.95f32.powi(e));
                loss[[]] * data.len() as f32
            })
            .reduce(|| 0.0, |acc, x| acc + x)
            // .fold(0.0, |acc, x| acc + x)
            / y.len() as f32;

        println!("{}", loss);
    }

    for i in 0..10 {
        let xs = Array2::from_shape_vec([1, 1], vec![i as f32 / 10.0])
            .unwrap()
            .into_ndarray();
        let ys = mlp.call(xs.into(), false);
        println!("{:?}", &*ys);
    }
}
