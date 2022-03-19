use ruzero::{
    functions::*,
    *,
};

fn main() {
    let x = backprop(scalar(2.0));
    let y = Mul.call(vec![x.clone(), x.clone()]);
    println!("{:?}", *y[0]);

    // let d = numerical_diff(&Square, &x);
    // println!("{}", *d);

    {
        let x = backprop(scalar(0.5)).named("x");
        let a = Mul.call(vec![x.clone(), x.clone()]);
        let b = Exp.call(a);
        let y = Mul.call(vec![b[0].clone(), b[0].clone()]);
        y[0].set_name("y");

        println!("{:?}", *y[0]);
        // y[0].backward(false, true);
        let gs = gradients(&y, &vec![x.clone()], true);
        println!("{:?}", gs[0][[]]); // 3.29

        let gs = gradients(&gs, &vec![x.clone()], false);
        // gx.backward(false, true);
        println!("{:?}", gs[0][[]]); // 13.18

        gs[0].set_name("x_grad");

        ruzero::export_dot::export_dot(&[gs[0].clone()], "graph.dot")
            .unwrap();

        // let f = std::fs::File::create("graph.dot").unwrap();
        // let mut w = std::io::BufWriter::new(f);
        // ruzero::export_dot::write_dot(&mut w, &x.get_grad().unwrap(), &mut |v| {
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
        let x = Variable::new(
            ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_tensor(),
        );
        let y = Sin.call(vec![x]);
        println!("{:?}", *y[0]);
        println!("{:?}", y[0].to_shape((3, 2, 1)));
    }
}
