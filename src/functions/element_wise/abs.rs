use crate::*;

pub fn abs(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new(x.map(|x| x.abs()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "abs",
        move |xs, _, gys| {
            vec![
                &gys[0]
                    * &ComputedNDA::new(
                        xs[0]
                            .map(|x| if *x >= 0.0 { 1.0 } else { -1.0 })
                            .into_ndarray(),
                    ),
            ]
        },
    );

    y
}

#[test]
fn test() {
    let x = backprop(ndarray::array![0., 1., -2., 3., -4.].into_ndarray());
    let y = abs(&x);
    assert_eq!(&*y, &ndarray::array![0., 1., 2., 3., 4.].into_ndarray());

    let grads = gradients(&[y], &[x.clone()], true);
    assert_eq!(
        &*grads[0],
        &ndarray::array![1., 1., -1., 1., -1.].into_ndarray()
    );
}
