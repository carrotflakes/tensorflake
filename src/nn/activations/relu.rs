use ndarray::Zip;

use crate::*;

pub fn relu(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((**x).map(|x| x.max(0.0)).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "relu",
        move |xs, _ys, gys| {
            let mut gx = (*gys[0]).to_owned();

            Zip::from(&mut gx).and(&xs[0].view()).for_each(|y, x| {
                if *x < 0.0 {
                    *y = 0.0
                }
            });
            vec![ComputedNDA::new(gx.into_ndarray())]
        },
    );

    y
}
