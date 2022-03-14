use ndarray::Array;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ruzero::{
    call,
    functions::{Add, Div, Matmul, Mul, Pow, Sub, SumTo},
    release_variables, scalar, Function, Variable, DISABLE_BACKPROP, ENABLE_BACKPROP,
};

fn main() {
    let x =
        Variable::<ENABLE_BACKPROP>::new(Array::random((3, 1), Uniform::new(0., 1.)).into_dyn());
    let y = call!(
        Add,
        call!(Mul, x, Variable::new(scalar(2.0))),
        Variable::new(Array::random((3, 1), Uniform::new(-0.01, 0.01)).into_dyn())
    );

    dbg!(&*x);
    dbg!(&*y);

    let mut w = Variable::new(ndarray::array![[0.0]].into_dyn());
    let mut b = Variable::new(ndarray::array![0.0].into_dyn());

    let predict =
        |w: Variable<ENABLE_BACKPROP>,
         b: Variable<ENABLE_BACKPROP>,
         x: Variable<ENABLE_BACKPROP>| call!(Add, call!(Matmul, x, w), b);

    for _ in 0..10 {
        let y_ = predict(w.clone(), b.clone(), x.clone());
        // dbg!(&*y_);

        let loss = mean_squared_error(y.clone(), y_.clone());
        dbg!(loss[[]]);

        loss.set_grad(Variable::<ENABLE_BACKPROP>::new(scalar(1.0)));
        loss.backward(false, false);

        let gw = w.get_grad::<ENABLE_BACKPROP>().unwrap();
        let gb = b.get_grad::<ENABLE_BACKPROP>().unwrap();
        // dbg!(&*gw);
        // dbg!(&*gb);

        let lr = 0.1;
        w = Variable::new(&*w - &*gw * lr);
        b = Variable::new(&*b - &*gb * lr);
        release_variables(&loss);
    }
}

fn mean_squared_error<const EB: bool>(x0: Variable<EB>, x1: Variable<EB>) -> Variable<EB> {
    call!(
        Div,
        call!(
            SumTo::new(vec![0, 1]),
            call!(Pow::new(2.0), call!(Sub, x0, x1))
        ),
        Variable::new(scalar(x0.shape().iter().sum::<usize>() as f32))
    )
}
