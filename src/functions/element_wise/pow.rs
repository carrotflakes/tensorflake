use ndarray::azip;

use crate::*;

pub fn pow(a: &ComputedNDA, b: &ComputedNDA) -> ComputedNDA {
    let mut y = NDArray::zeros(a.shape());
    azip!((y in &mut y, a in &**a, b in &**b) *y = a.powf(*b));
    let y = Computed::new(y);

    chain(
        &[a.clone(), b.clone()],
        &[y.clone()],
        false,
        "pow",
        move |xs, ys, gys| {
            let ga =
                &gys[0] * &pow(&xs[0], &(&xs[1] - &Computed::new(scalar(1.0)))) * xs[1].clone();
            let gb = &gys[0] * &ys[0] * xs[0].log();
            vec![ga, gb]
        },
    );

    y
}

pub fn pow_const(x: &ComputedNDA, a: f32) -> ComputedNDA {
    let y = ComputedNDA::new((**x).map(|x| x.powf(a)).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "pow_const",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].pow_const(a - 1.0) * ComputedNDA::new(scalar(a));
            vec![gx]
        },
    );

    y
}

#[test]
fn test_pow() {
    let a = backprop(scalar(5.0));
    let y = pow_const(&a, 2.0);
    assert_eq!(y[[]], 25.0);

    let grads = gradients(&[y], &[a.clone()], false);
    assert_eq!(*grads[0], scalar(10.0));

    let b = backprop(scalar(2.0));
    let y = pow(&a, &b);
    assert_eq!(y[[]], 25.0);

    let grads = gradients(&[y], &[a.clone(), b.clone()], false);
    assert_eq!(*grads[0], scalar(10.0));
    assert_eq!(*grads[1], scalar(25.0) * &*a.log());
}
