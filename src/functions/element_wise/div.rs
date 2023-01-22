use crate::{functions::sum_axes_to_desire, *};

pub fn div(a: &ComputedNDA, b: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((&**a / &**b).into_ndarray());

    chain(
        &[a.clone(), b.clone()],
        &[y.clone()],
        false,
        "div",
        |xs, _ys, gys| {
            let mut gx0 = &gys[0] / &xs[0];

            let mut gx1 = &gys[0] * &(-&xs[0] / xs[1].pow_const(2.0));

            // fit shape
            if xs[0].shape() != gx0.shape() {
                gx0 = gx0.sum(sum_axes_to_desire(gx0.shape(), xs[0].shape()), false);
            }

            if xs[1].shape() != gx1.shape() {
                gx1 = gx1.sum(sum_axes_to_desire(gx1.shape(), xs[0].shape()), false);
            }

            vec![gx0, gx1]
        },
    );

    y
}
