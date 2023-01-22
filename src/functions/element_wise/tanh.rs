use crate::*;

pub fn tanh(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new(x.map(|x| x.tanh()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "tanh",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(ComputedNDA::new(scalar(1.0)) - ys[0].pow_const(2.0));
            vec![gx]
        },
    );

    y
}
