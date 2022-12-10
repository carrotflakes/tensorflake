use crate::*;

pub fn t(x: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((**x).t().into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "t",
        move |_xs, _ys, gys| {
            let gx = gys[0].t();
            vec![gx]
        },
    );

    y
}

#[test]
fn test() {
    {
        let x = ComputedNDA::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = t(&x);
        assert_eq!(&y.shape(), &[3, 2]);
    }

    {
        let x = ComputedNDA::new(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
        let y = t(&x);
        assert_eq!(&y.shape(), &[3, 2, 1]);
    }
}
