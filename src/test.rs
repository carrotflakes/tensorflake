use crate::{
    functions::{Add, Mul, Pow, Sub},
    release_variables, Function, Tensor, Variable,
};

fn scalar(x: f32) -> Tensor {
    ndarray::arr0(x).into_dyn()
}

#[test]
fn test_add_mul() {
    let a = Variable::<true>::new(scalar(3.0));
    let b = Variable::new(scalar(2.0));
    let c = Variable::new(scalar(1.0));

    let ys = Add.call(vec![
        Mul.call(vec![a.clone(), b.clone()]).pop().unwrap(),
        c.clone(),
    ]);
    assert_eq!(*ys[0], scalar(7.0));

    ys[0].set_grad(Variable::<true>::new(scalar(1.0)));
    ys[0].backward(false, false);
    assert_eq!(
        a.get_grad::<false>().map(|v| (*v).clone()),
        Some(scalar(2.0))
    );
    assert_eq!(
        b.get_grad::<false>().map(|v| (*v).clone()),
        Some(scalar(3.0))
    );
}

#[test]
fn test_sphere() {
    let x = Variable::<true>::new(scalar(1.0));
    let y = Variable::new(scalar(1.0));

    let ys = Add.call(vec![
        Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
        Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
    ]);
    assert_eq!(*ys[0], scalar(2.0));

    ys[0].set_grad(Variable::<true>::new(scalar(1.0)));
    ys[0].backward(false, false);
    assert_eq!(
        x.get_grad::<false>().map(|v| (*v).clone()),
        Some(scalar(2.0))
    );
    assert_eq!(
        y.get_grad::<false>().map(|v| (*v).clone()),
        Some(scalar(2.0))
    );
}

#[test]
fn test_matyas() {
    let x = Variable::<true>::new(scalar(1.0));
    let y = Variable::new(scalar(1.0));

    let ys = Sub.call(vec![
        Mul.call(vec![
            Variable::new(scalar(0.26)),
            Add.call(vec![
                Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
                Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
            ])
            .pop()
            .unwrap(),
        ])
        .pop()
        .unwrap(),
        Mul.call(vec![Variable::new(scalar(0.48)), x.clone(), y.clone()])
            .pop()
            .unwrap(),
    ]);

    ys[0].set_grad(Variable::<true>::new(scalar(1.0)));
    ys[0].backward(false, false);
    assert!(
        (&*x.get_grad::<false>().unwrap() - 0.04)
            .iter()
            .next()
            .unwrap()
            .abs()
            < 1e-6
    );
    assert!(
        (&*y.get_grad::<false>().unwrap() - 0.04)
            .iter()
            .next()
            .unwrap()
            .abs()
            < 1e-6
    );
}

fn rosenbrock<const ENABLE_BACKPROP: bool>(
    a: Variable<ENABLE_BACKPROP>,
    b: Variable<ENABLE_BACKPROP>,
) -> Variable<ENABLE_BACKPROP> {
    Add.call(vec![
        Mul.call(vec![
            Variable::new(scalar(100.0)),
            Pow::new(2.0)
                .call(vec![Sub
                    .call(vec![
                        b.clone(),
                        Pow::new(2.0).call(vec![a.clone()]).pop().unwrap(),
                    ])
                    .pop()
                    .unwrap()])
                .pop()
                .unwrap(),
        ])
        .pop()
        .unwrap(),
        Pow::new(2.0)
            .call(vec![Sub
                .call(vec![a.clone(), Variable::new(scalar(1.0))])
                .pop()
                .unwrap()])
            .pop()
            .unwrap(),
    ])
    .pop()
    .unwrap()
}

#[test]
fn test_rosenbrock() {
    let a = Variable::<true>::new(scalar(0.0));
    let b = Variable::new(scalar(2.0));

    let y = rosenbrock(a.clone(), b.clone());

    y.set_grad(Variable::<true>::new(scalar(1.0)));
    y.backward(false, false);
    assert!(
        (&*a.get_grad::<false>().unwrap() - -2.0)
            .iter()
            .next()
            .unwrap()
            .abs()
            < 1e-6
    );
    assert!(
        (&*b.get_grad::<false>().unwrap() - 400.0)
            .iter()
            .next()
            .unwrap()
            .abs()
            < 1e-6
    );
}

#[test]
fn test_rosenbrock_sgd() {
    let mut a = Variable::<true>::new(scalar(0.0));
    let mut b = Variable::new(scalar(2.0));
    let lr = 0.001;

    for _ in 0..50000 {
        // dbg!((a.get_data()[0], b.get_data()[0]));
        let y = rosenbrock(a.clone(), b.clone());

        y.set_grad(Variable::<true>::new(scalar(1.0)));
        y.backward(false, false);

        a = Variable::new(&*a - &*a.get_grad::<false>().unwrap() * lr);
        b = Variable::new(&*b - &*b.get_grad::<false>().unwrap() * lr);
        release_variables(&y);
    }

    assert!((&*a - 1.0).iter().next().unwrap().abs() < 1e-3);
    assert!((&*b - 1.0).iter().next().unwrap().abs() < 1e-3);
}

macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.clone()),*]).pop().unwrap()
    };
}

#[test]
fn second_order_differentia() {
    let x = Variable::<true>::new(scalar(2.0));

    let y = call!(
        Sub,
        call!(Pow::new(4.0), x),
        call!(Mul, Variable::new(scalar(2.0)), call!(Pow::new(2.0), x))
    );
    assert_eq!(*y, scalar(8.0));

    y.set_grad(Variable::<true>::new(scalar(1.0)));
    y.backward(false, true);
    assert_eq!(*x.get_grad::<true>().unwrap(), scalar(24.0));

    let gx = x.get_grad::<true>().unwrap();
    x.clear_grad();
    gx.set_grad(Variable::<true>::new(scalar(1.0)));
    gx.backward(false, false);
    assert_eq!(*x.get_grad::<false>().unwrap(), scalar(44.0));
}

#[test]
fn test_() {
    let x = Variable::<true>::new(scalar(2.0));
    let xs = vec![x];
    let ys: &Vec<Variable<false>> = unsafe { std::mem::transmute(&xs) };
    dbg!(&*xs[0]);
    dbg!(&*ys[0]);
}
