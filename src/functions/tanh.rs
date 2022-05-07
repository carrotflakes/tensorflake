use crate::*;

pub fn tanh(x: &Computed) -> Computed {
    let y = Computed::new(x.map(|x| x.tanh()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "tanh",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &(Computed::new(scalar(1.0)) - ys[0].pow(2.0));
            vec![gx]
        },
    );

    y
}
