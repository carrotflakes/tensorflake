use crate::functions::*;
use crate::*;

pub fn matmul_add(x0: &Computed, x1: &Computed, x2: &Computed) -> Computed {
    let y = matmul::forward(&x0, &x1);
    let y = y + &**x2;
    let y = Computed::new(y);

    chain(
        &[x0.clone(), x1.clone(), x2.clone()],
        &[y.clone()],
        false,
        "matmul_add",
        move |xs, _, gys| {
            let gx0 = matmul(&gys[0], &mat_transpose(&xs[1]));
            let gx1 = matmul(&mat_transpose(&xs[0]), &gys[0]);

            let mut gx2 = gys[0].clone();

            // fit shape
            if xs[2].shape() != gx2.shape() {
                gx2 = sum(&gx2, sum_axes_to_desire(gx2.shape(), xs[2].shape()), false);
            }

            vec![gx0.into(), gx1.into(), gx2]
        },
    );

    y
}

#[test]
fn test() {
    let a = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
    let b = backprop(ndarray::array![[1., 2.], [3., 4.], [5., 6.]].into_ndarray());
    let c = backprop(ndarray::array![1., 2.].into_ndarray());
    let y = matmul_add(&a, &b, &c);
    dbg!(&*y);
    let t = add(&matmul(&a, &b), &c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);

    let a = backprop(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
    let b = backprop(ndarray::array![[[1., 2.], [3., 4.], [5., 6.]]].into_ndarray());
    let c = backprop(ndarray::array![[1., 2.]].into_ndarray());
    let y = matmul_add(&a, &b, &c);
    dbg!(&*y);
    let t = add(&matmul(&a, &b), &c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);

    let a = backprop(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
    let b = backprop(ndarray::array![[[1.], [3.], [5.]]].into_ndarray());
    let c = backprop(ndarray::array![[1.]].into_ndarray());
    let y = matmul_add(&a, &b, &c);
    dbg!(&*y);
    let t = add(&matmul(&a, &b), &c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);
}
