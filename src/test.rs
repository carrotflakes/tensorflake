use crate::{
    functions::{Mul, Pow, Sub, Sum},
    Function, Variable, release_variables,
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
    ys[0].backward(false);
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
    ys[0].backward(false);
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
    ys[0].backward(false);
    assert!((x.get_grad().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
    assert!((y.get_grad().unwrap().get_data()[0] - 0.04).abs() < 1e-6);
}

fn rosenbrock(a: Variable, b: Variable) -> Variable {
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
    let a = Variable::new(0.0.into());
    let b = Variable::new(2.0.into());

    let y = rosenbrock(a.clone(), b.clone());

    y.set_grad(Variable::new(1.0.into()));
    y.backward(false);
    assert!((a.get_grad().unwrap().get_data()[0] - -2.0).abs() < 1e-6);
    assert!((b.get_grad().unwrap().get_data()[0] - 400.0).abs() < 1e-6);
}

#[test]
fn test_rosenbrock_sgd() {
    let mut a = Variable::new(0.0.into());
    let mut b = Variable::new(2.0.into());
    let lr = 0.001;

    for _ in 0..50000 {
        // dbg!((a.get_data()[0], b.get_data()[0]));
        let y = rosenbrock(a.clone(), b.clone());
        
        y.set_grad(Variable::new(1.0.into()));
        y.backward(false);

        a = Variable::new((a.data[0] - lr * a.get_grad().unwrap().data[0]).into());
        b = Variable::new((b.data[0] - lr * b.get_grad().unwrap().data[0]).into());
        release_variables(&y);
    }

    assert!((a.get_data()[0] - 1.0).abs() < 1e-3);
    assert!((b.get_data()[0] - 1.0).abs() < 1e-3);
}
