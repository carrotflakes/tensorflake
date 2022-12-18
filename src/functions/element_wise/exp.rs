use crate::*;

pub fn exp(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((**x).map(|x| x.exp()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "exp",
        move |_xs, ys, gys| {
            let gx = &gys[0] * &ys[0];
            vec![gx]
        },
    );

    y
}
