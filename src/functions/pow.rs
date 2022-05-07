use crate::*;

pub fn pow(x: &Computed, a: f32) -> Computed {
    let y = Computed::new((**x).map(|x| x.powf(a)).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "pow",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].pow(a - 1.0) * Computed::new(scalar(a));
            vec![gx]
        },
    );

    y
}

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

#[test]
fn test_pow() {
    let a = backprop(scalar(5.0));
    let y = pow(&a, 2.0);
    assert_eq!(y[[]], 25.0);

    let grads = gradients(&[y], &[a.clone()], false);
    assert_eq!(*grads[0], scalar(10.0));
}
