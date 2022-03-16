use ndarray::{Array, array};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ruzero::{functions::*, nn::*, *};

fn main() {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let n = 100;

    let x = Variable::<ENABLE_BACKPROP>::new(
        Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_dyn(),
    )
    .named("x");
    let y = call!(
        Add,
        call!(Sin, call!(Mul, x, Variable::new(scalar(2.0 * 3.14)))),
        Variable::new(Array::random_using((n, 1), Uniform::new(0.0, 1.0), &mut rng).into_dyn())
    )
    .named("y");

    let mut l1 = Layer::new(1, 10);
    let mut l2 = Layer::new(10, 1);

    // let mut ll = 1000.0;

    for i in 0..10000 {
        let h = l1.forward(x.clone());
        let h = sigmoid_simple(h).named("hidden");
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

        loss.backward(false, false);

        // dbg!(&*l1.w.get_grad::<ENABLE_BACKPROP>().unwrap());
        // dbg!(&*l1.b.get_grad::<ENABLE_BACKPROP>().unwrap());
        let lr = 0.1;
        l1.update(lr);
        l2.update(lr);
    }
    for i in 0..20 {
        let x = Variable::<ENABLE_BACKPROP>::new(
            array![[i as f32 / 20.0]].into_dyn()
        );
        let h = l1.forward(x);
        let h = sigmoid_simple(h).named("hidden");
        let y_ = l2.forward(h);
        println!("{}", &*y_);
    }
    println!("end");
}

fn mean_squared_error<const EB: bool>(x0: Variable<EB>, x1: Variable<EB>) -> Variable<EB> {
    let x = call!(Pow::new(2.0), call!(Sub, x0, x1));
    call!(
        Div,
        call!(SumTo::new((0..x.ndim()).collect()), x),
        Variable::new(scalar(x.shape().iter().product::<usize>() as f32))
    )
}

pub struct Layer {
    pub w: Variable<ENABLE_BACKPROP>,
    pub b: Variable<ENABLE_BACKPROP>,
}

impl Layer {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            w: Variable::new(Array::random((input, output), Uniform::new(0., 0.01)).into_dyn())
                .named("param"),
            b: Variable::new(Array::zeros(output).into_dyn()).named("param"),
        }
    }

    pub fn forward(&self, x: Variable<ENABLE_BACKPROP>) -> Variable<ENABLE_BACKPROP> {
        call!(Add, call!(Matmul, x, self.w), self.b).named("return")
    }

    pub fn update(&mut self, lr: f32) {
        self.w = Variable::new(&*self.w - &*self.w.get_grad::<ENABLE_BACKPROP>().unwrap() * lr);
        self.b = Variable::new(&*self.b - &*self.b.get_grad::<ENABLE_BACKPROP>().unwrap() * lr);
    }
}

fn graph(vars: &[Variable<ENABLE_BACKPROP>], name: impl ToString) {
    let f = std::fs::File::create(name.to_string() + ".dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    ruzero::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
