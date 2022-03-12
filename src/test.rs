use crate::{
    functions::{Mul, Pow, Sub, Sum},
    release_variables, Function, Variable,
};

#[test]
fn test_add_mul() {
    let a = Variable::<true>::new(3.0.into());
    let b = Variable::new(2.0.into());
    let c = Variable::new(1.0.into());

    let ys = Sum.call(vec![
        Mul.call(vec![a.clone(), b.clone()]).pop().unwrap(),
        c.clone(),
    ]);
    assert_eq!(*ys[0], 7.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward::<false>(false);
    assert_eq!(
        a.get_grad::<false>().map(|v| (*v).clone()),
        Some(2.0.into())
    );
    assert_eq!(
        b.get_grad::<false>().map(|v| (*v).clone()),
        Some(3.0.into())
    );
}

#[test]
fn test_sphere() {
    let x = Variable::<true>::new(1.0.into());
    let y = Variable::new(1.0.into());

    let ys = Sum.call(vec![
        Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
        Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
    ]);
    assert_eq!(*ys[0], 2.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward::<false>(false);
    assert_eq!(
        x.get_grad::<false>().map(|v| (*v).clone()),
        Some(2.0.into())
    );
    assert_eq!(
        y.get_grad::<false>().map(|v| (*v).clone()),
        Some(2.0.into())
    );
}

#[test]
fn test_matyas() {
    let x = Variable::<true>::new(1.0.into());
    let y = Variable::new(1.0.into());

    let ys = Sub.call(vec![
        Mul.call(vec![
            Variable::new(0.26.into()),
            Sum.call(vec![
                Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
                Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
            ])
            .pop()
            .unwrap(),
        ])
        .pop()
        .unwrap(),
        Mul.call(vec![Variable::new(0.48.into()), x.clone(), y.clone()])
            .pop()
            .unwrap(),
    ]);

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward::<false>(false);
    assert!((x.get_grad::<false>().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
    assert!((y.get_grad::<false>().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
}

fn rosenbrock<const ENABLE_BACKPROP: bool>(
    a: Variable<ENABLE_BACKPROP>,
    b: Variable<ENABLE_BACKPROP>,
) -> Variable<ENABLE_BACKPROP> {
    Sum.call(vec![
        Mul.call(vec![
            Variable::new(100.0.into()),
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
                .call(vec![a.clone(), Variable::new(1.0.into())])
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
    let a = Variable::<true>::new(0.0.into());
    let b = Variable::new(2.0.into());

    let y = rosenbrock(a.clone(), b.clone());

    y.set_grad(Variable::new(1.0.into()));
    y.backward::<false>(false);
    assert!((a.get_grad::<false>().unwrap().get_data()[0] - -2.0).abs() < 1e-6);
    assert!((b.get_grad::<false>().unwrap().get_data()[0] - 400.0).abs() < 1e-6);
}

#[test]
fn test_rosenbrock_sgd() {
    let mut a = Variable::<true>::new(0.0.into());
    let mut b = Variable::new(2.0.into());
    let lr = 0.001;

    for _ in 0..50000 {
        // dbg!((a.get_data()[0], b.get_data()[0]));
        let y = rosenbrock(a.clone(), b.clone());

        y.set_grad(Variable::new(1.0.into()));
        y.backward::<false>(false);

        a = Variable::new((a.data[0] - lr * a.get_grad::<false>().unwrap().data[0]).into());
        b = Variable::new((b.data[0] - lr * b.get_grad::<false>().unwrap().data[0]).into());
        release_variables(&y);
    }

    assert!((a.get_data()[0] - 1.0).abs() < 1e-3);
    assert!((b.get_data()[0] - 1.0).abs() < 1e-3);
}

macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es),*]).pop().unwrap()
    };
}

#[test]
fn second_order_differentia() {
    let x = Variable::<true>::new(2.0.into());

    let y = call!(
        Sub,
        // call!(Pow::new(4.0),
        //     x.clone()
        // ),
        call!(Mul, x.clone(), x.clone(), x.clone(), x.clone()),
        call!(
            Mul,
            Variable::new(2.0.into()),
            // call!(Pow::new(2.0),
            //     x.clone()
            // )
            call!(Mul, x.clone(), x.clone())
        )
    );
    assert_eq!(y.data[0], 8.0.into());

    y.set_grad(Variable::new(1.0.into()));
    y.backward::<true>(false);
    assert_eq!(x.get_grad::<true>().unwrap().data[0], 24.0.into());

    let gx = x.get_grad::<true>().unwrap();
    x.clear_grad();
    gx.set_grad(Variable::new(1.0.into()));
    gx.backward::<false>(false);
    assert_eq!(x.get_grad::<false>().unwrap().data[0], 44.0.into());
}
