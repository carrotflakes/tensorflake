use crate::*;

pub fn sin(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.sin()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sin",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].cos();
            vec![gx]
        },
    );

    y
}

pub fn cos(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.cos()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "cos",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &-&xs[0].sin();
            vec![gx]
        },
    );

    y
}
