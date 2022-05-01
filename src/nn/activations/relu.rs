use crate::*;

pub fn relu(x: &Computed) -> Computed {
    let y = Computed::new((**x).map(|x| x.max(0.0)).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "relu",
        move |xs, _ys, gys| {
            let gx = &gys[0]
                * &Computed::new(
                    xs[0]
                        .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                        .into_ndarray(),
                );
            vec![gx]
        },
    );

    y
}
