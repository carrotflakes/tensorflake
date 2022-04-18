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
            let gx0 = Matmul.call(vec![
                gys[0].clone(),
                MatTranspose.call(vec![xs[1].clone()])[0].clone(),
            ])[0]
                .clone();
            let gx1 = Matmul.call(vec![
                MatTranspose.call(vec![xs[0].clone()])[0].clone(),
                gys[0].clone(),
            ])[0]
                .clone();

            let mut gx2 = gys[0].clone();

            // fit shape
            if xs[2].shape() != gx2.shape() {
                gx2 = call!(
                    Sum::new(sum_axes_to_desire(gx2.shape(), xs[2].shape()), false),
                    gx2
                );
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
    let t = call!(Add, call!(Matmul, a, b), c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);

    let a = backprop(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
    let b = backprop(ndarray::array![[[1., 2.], [3., 4.], [5., 6.]]].into_ndarray());
    let c = backprop(ndarray::array![[1., 2.]].into_ndarray());
    let y = matmul_add(&a, &b, &c);
    dbg!(&*y);
    let t = call!(Add, call!(Matmul, a, b), c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);

    let a = backprop(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
    let b = backprop(ndarray::array![[[1.], [3.], [5.]]].into_ndarray());
    let c = backprop(ndarray::array![[1.]].into_ndarray());
    let y = matmul_add(&a, &b, &c);
    dbg!(&*y);
    let t = call!(Add, call!(Matmul, a, b), c);
    assert_eq!(&*y, &*t);

    gradients(&[y], &[a, b, c], false);
}
