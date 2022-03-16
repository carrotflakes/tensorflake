use ruzero::{
    functions::{Exp, Mul, Sin},
    scalar, Function, Variable, DISABLE_BACKPROP, ENABLE_BACKPROP,
};

fn main() {
    let x = Variable::<DISABLE_BACKPROP>::new(scalar(2.0));
    let y = Mul.call(vec![x.clone(), x.clone()]);
    println!("{:?}", *y[0]);

    // let d = numerical_diff(&Square, &x);
    // println!("{}", *d);

    {
        let x = Variable::new(scalar(0.5)).named("x");
        let a = Mul.call(vec![x.clone(), x.clone()]);
        let b = Exp.call(a);
        let y = Mul.call(vec![b[0].clone(), b[0].clone()]);
        y[0].set_name("y");

        println!("{:?}", *y[0]);
        y[0].backward(false, true);
        println!("{:?}", *x.get_grad::<DISABLE_BACKPROP>().unwrap()); // 3.29

        let gx = x.get_grad::<ENABLE_BACKPROP>().unwrap().clone();
        x.clear_grad();
        gx.backward(false, true);
        println!("{:?}", *x.get_grad::<DISABLE_BACKPROP>().unwrap()); // 13.18

        x.get_grad::<DISABLE_BACKPROP>().unwrap().set_name("x_grad");

        ruzero::export_dot::export_dot(&[x.get_grad::<ENABLE_BACKPROP>().unwrap()], "graph.dot")
            .unwrap();

        // let f = std::fs::File::create("graph.dot").unwrap();
        // let mut w = std::io::BufWriter::new(f);
        // ruzero::export_dot::write_dot(&mut w, &x.get_grad::<ENABLE_BACKPROP>().unwrap(), &mut |v| {
        //     format!("{} {}", v.get_name(), (*v).to_string())
        // })
        // .unwrap();

        // ruzero::release_variables(&y[0]);
        // y.set_grad(1.0);
        // b.set_grad(*Square.backward(&b, &Variable::new(y.get_grad().unwrap())));
        // a.set_grad(*Exp.backward(&a, &Variable::new(b.get_grad().unwrap())));
        // x.set_grad(*Square.backward(&x, &Variable::new(a.get_grad().unwrap())));
        // println!("{:?}", x.get_grad());
    }

    {
        let x = Variable::<ENABLE_BACKPROP>::new(
            ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
        );
        let y = Sin.call(vec![x]);
        println!("{:?}", *y[0]);
        println!("{:?}", y[0].to_shape((3, 2, 1)));
    }
}
