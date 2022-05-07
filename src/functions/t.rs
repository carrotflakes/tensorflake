use crate::*;

pub fn t(x: &Computed) -> Computed {
    let y = Computed::new((**x).t().into_ndarray());

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
        let x = Computed::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = t(&x);
        assert_eq!(&y.shape(), &[3, 2]);
    }

    {
        let x = Computed::new(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
        let y = t(&x);
        assert_eq!(&y.shape(), &[3, 2, 1]);
    }
}
