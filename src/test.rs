use crate::{
    functions::{Mul, Pow, Sub, Sum},
    Function, Variable,
};

#[test]
fn test_add_mul() {
    let a = Variable::new(3.0.into());
    let b = Variable::new(2.0.into());
    let c = Variable::new(1.0.into());

    let ys = Sum.call(vec![
        Mul.call(vec![a.clone(), b.clone()]).pop().unwrap(),
        c.clone(),
    ]);
    assert_eq!(*ys[0], 7.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward();
    assert_eq!(a.get_grad().map(|v| (*v).clone()), Some(2.0.into()));
    assert_eq!(b.get_grad().map(|v| (*v).clone()), Some(3.0.into()));
}

#[test]
fn test_sphere() {
    let x = Variable::new(1.0.into());
    let y = Variable::new(1.0.into());

    let ys = Sum.call(vec![
        Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
        Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
    ]);
    assert_eq!(*ys[0], 2.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward();
    assert_eq!(x.get_grad().map(|v| (*v).clone()), Some(2.0.into()));
    assert_eq!(y.get_grad().map(|v| (*v).clone()), Some(2.0.into()));
}

#[test]
fn test_matyas() {
    let x = Variable::new(1.0.into());
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
    ys[0].backward();
    assert!((x.get_grad().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
    assert!((y.get_grad().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
}

#[test]
fn test_rosenbrock() {
    let a = Variable::new(0.0.into());
    let b = Variable::new(2.0.into());

    let y = Sum
        .call(vec![
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
        .unwrap();

    y.set_grad(Variable::new(1.0.into()));
    y.backward();
    assert!((a.get_grad().unwrap().get_data()[0] - -2.0).abs() < 1e-6);
    assert!((b.get_grad().unwrap().get_data()[0] - 400.0).abs() < 1e-6);
}
