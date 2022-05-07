use crate::*;

pub fn neg(x: &Computed) -> Computed {
    let y = Computed::new((-&**x).into_ndarray());

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
