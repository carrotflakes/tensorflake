use crate::*;

pub fn sigmoid(x: &Computed) -> Computed {
    let y = Computed::new(x.map(|x| (x * 0.5).tanh() * 0.5 + 0.5).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sigmoid",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(&Computed::new(scalar(1.0)) - &ys[0]) * ys[0].clone();
            vec![gx]
        },
    );

    y
}

pub fn naive_sigmoid(x: Computed) -> Computed {
    Computed::new(scalar(1.0)) / (Computed::new(scalar(1.0)) + (-x).exp())
}
