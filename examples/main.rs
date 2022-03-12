use ruzero::{
    functions::{Exp, Mul},
    Function, Variable,
};

fn main() {
    let x = Variable::<false>::new(2.0.into());
    let y = Mul.call(vec![x.clone(), x.clone()]);
    println!("{:?}", *y[0]);

    // let d = numerical_diff(&Square, &x);
    // println!("{}", *d);

    {
        let x = Variable::<true>::new(0.5.into());
        let a = Mul.call(vec![x.clone(), x.clone()]);
        let b = Exp.call(a);
        let y = Mul.call(vec![b[0].clone(), b[0].clone()]);
        println!("{:?}", *y[0]);
        y[0].set_grad(Variable::<true>::new(1.0.into()));
        y[0].backward(false, true);
        println!("{:?}", *x.get_grad::<false>().unwrap()); // 3.29

        let gx = x.get_grad::<true>().unwrap().clone();
        x.clear_grad();
        gx.set_grad(Variable::<true>::new(1.0.into()));
        gx.backward(false, false);
        println!("{:?}", *x.get_grad::<false>().unwrap()); // 13.18

        // ruzero::release_variables(&y[0]);
        // y.set_grad(1.0);
        // b.set_grad(*Square.backward(&b, &Variable::new(y.get_grad().unwrap())));
        // a.set_grad(*Exp.backward(&a, &Variable::new(b.get_grad().unwrap())));
        // x.set_grad(*Square.backward(&x, &Variable::new(a.get_grad().unwrap())));
        // println!("{:?}", x.get_grad());
    }
}
