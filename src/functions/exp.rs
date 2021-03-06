use crate::*;

pub fn exp(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.exp()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "exp",
        move |xs, _ys, gys| {
            let gx = &gys[0] * &xs[0].exp();
            vec![gx]
        },
    );

    y
}
