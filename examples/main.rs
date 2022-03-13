use ruzero::{
    functions::{Exp, Mul, Sin},
    Function, Variable, Tensor, DISABLE_BACKPROP, ENABLE_BACKPROP,
};

fn main() {
    let x = Variable::<DISABLE_BACKPROP>::new(2.0.into());
    let y = Mul.call(vec![x.clone(), x.clone()]);
    println!("{:?}", *y[0]);

    // let d = numerical_diff(&Square, &x);
    // println!("{}", *d);

    {
        let x = Variable::new(0.5.into());
        let a = Mul.call(vec![x.clone(), x.clone()]);
        let b = Exp.call(a);
        let y = Mul.call(vec![b[0].clone(), b[0].clone()]);
        println!("{:?}", *y[0]);
        y[0].set_grad(Variable::<ENABLE_BACKPROP>::new(1.0.into()));
        y[0].backward(false, true);
        println!("{:?}", *x.get_grad::<DISABLE_BACKPROP>().unwrap()); // 3.29

        let gx = x.get_grad::<ENABLE_BACKPROP>().unwrap().clone();
        x.clear_grad();
        gx.set_grad(Variable::<ENABLE_BACKPROP>::new(1.0.into()));
        gx.backward(false, false);
        println!("{:?}", *x.get_grad::<DISABLE_BACKPROP>().unwrap()); // 13.18

        // ruzero::release_variables(&y[0]);
        // y.set_grad(1.0);
        // b.set_grad(*Square.backward(&b, &Variable::new(y.get_grad().unwrap())));
        // a.set_grad(*Exp.backward(&a, &Variable::new(b.get_grad().unwrap())));
        // x.set_grad(*Square.backward(&x, &Variable::new(a.get_grad().unwrap())));
        // println!("{:?}", x.get_grad());
    }

    {
        let x = Variable::<ENABLE_BACKPROP>::new(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let y = Sin.call(vec![x]);
        println!("{:?}", *y[0]);
        println!("{:?}", y[0].reshape(&[3, 2, 1]));
    }
}
