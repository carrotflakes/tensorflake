use crate::*;

pub fn neg(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((-&**x).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "neg",
        move |_xs, _ys, gys| {
            let gx = -&gys[0];
            vec![gx]
        },
    );

    y
}
