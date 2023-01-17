use ndarray_rand::rand_distr::Uniform;
use tensorflake::{
    initializers::{
        random_initializer::RandomInitializer, with_optimizer::InitializerWithOptimizer, Scope,
    },
    nn::Linear,
    optimizers,
};

fn main() {
    let init = InitializerWithOptimizer::new(
        RandomInitializer::new(Uniform::new(0., 0.01)),
        optimizers::SGD::new(0.01),
    );

    let linear = Linear::new(10, 10, init.scope("w"), Some(init.scope("b")));

    dbg!(&*linear.b.as_ref().unwrap().get());

    let file = "linear.bincode";

    // save
    {
        let mut w = std::fs::File::create(file).unwrap();
        bincode::serialize_into(&mut w, &linear).unwrap();
    }

    // load
    {
        let mut r = std::fs::File::open(file).unwrap();
        let linear: Linear = bincode::deserialize_from(&mut r).unwrap();
        dbg!(&*linear.b.as_ref().unwrap().get());
    }
}
