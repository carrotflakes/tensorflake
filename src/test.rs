use crate::{
    functions::{Add, Mul, Pow, Sub},
    *,
};

#[test]
fn test_add_mul() {
    let a = backprop(scalar(3.0));
    let b = backprop(scalar(2.0));
    let c = backprop(scalar(1.0));

    let ys = Add.call(vec![
        Mul.call(vec![a.clone(), b.clone()]).pop().unwrap(),
        c.clone(),
    ]);
    assert_eq!(*ys[0], scalar(7.0));

    let grads = gradients(&ys, &vec![a.clone(), b.clone()], false);
    assert_eq!(&*grads[0], scalar(2.0));
    assert_eq!(&*grads[1], scalar(3.0));
}

#[test]
fn test_sphere() {
    let x = backprop(scalar(1.0));
    let y = backprop(scalar(1.0));

    let ys = Add.call(vec![
        Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
        Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
    ]);
    assert_eq!(*ys[0], scalar(2.0));

    let grads = gradients(&ys, &vec![x.clone(), y.clone()], false);
    assert_eq!(&*grads[0], scalar(2.0));
    assert_eq!(&*grads[1], scalar(2.0));
}

#[test]
fn test_matyas() {
    let x = backprop(scalar(1.0));
    let y = backprop(scalar(1.0));

    let ys = Sub.call(vec![
        Mul.call(vec![
            Tensor::new(scalar(0.26)),
            Add.call(vec![
                Pow::new(2.0).call(vec![x.clone()]).pop().unwrap(),
                Pow::new(2.0).call(vec![y.clone()]).pop().unwrap(),
            ])
            .pop()
            .unwrap(),
        ])
        .pop()
        .unwrap(),
        Mul.call(vec![Tensor::new(scalar(0.48)), x.clone(), y.clone()])
            .pop()
            .unwrap(),
    ]);

    let grads = gradients(&ys, &vec![x.clone(), y.clone()], false);
    assert!((&*grads[0] - 0.04).iter().next().unwrap().abs() < 1e-6);
    assert!((&*grads[1] - 0.04).iter().next().unwrap().abs() < 1e-6);
}

fn rosenbrock(a: Tensor, b: Tensor) -> Tensor {
    Add.call(vec![
        Mul.call(vec![
            Tensor::new(scalar(100.0)),
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
                .call(vec![a.clone(), Tensor::new(scalar(1.0))])
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
    let a = backprop(scalar(0.0));
    let b = backprop(scalar(2.0));

    let y = rosenbrock(a.clone(), b.clone());

    let grads = gradients(&vec![y], &vec![a.clone(), b.clone()], false);
    assert!((&*grads[0] - -2.0).iter().next().unwrap().abs() < 1e-6);
    assert!((&*grads[1] - 400.0).iter().next().unwrap().abs() < 1e-6);
}

#[test]
fn test_rosenbrock_sgd() {
    let mut a = backprop(scalar(0.0));
    let mut b = backprop(scalar(2.0));
    let lr = 0.001;

    for _ in 0..50000 {
        // dbg!((a.get_data()[0], b.get_data()[0]));
        let y = rosenbrock(a.clone(), b.clone());

        let grads = gradients(&vec![y], &vec![a.clone(), b.clone()], false);
        a = backprop((&*a - &*grads[0] * lr).into_ndarray());
        b = backprop((&*b - &*grads[1] * lr).into_ndarray());
    }

    assert!((&*a - 1.0).iter().next().unwrap().abs() < 1e-3);
    assert!((&*b - 1.0).iter().next().unwrap().abs() < 1e-3);
}

#[test]
fn test_second_order_differentia() {
    let x = backprop(scalar(2.0));

    let y = call!(
        Sub,
        call!(Pow::new(4.0), x),
        call!(Mul, Tensor::new(scalar(2.0)), call!(Pow::new(2.0), x))
    );
    assert_eq!(*y, scalar(8.0));

    let grads = gradients(&vec![y.clone()], &vec![x.clone()], true);
    assert_eq!(grads[0][[]], 24.0);

    let grads = gradients(&vec![grads[0].clone()], &vec![x], false);
    assert_eq!(grads[0][[]], 44.0);
}
