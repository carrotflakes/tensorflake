use crate::*;

pub fn reshape(x: &Computed, shape: impl Into<Vec<usize>>) -> Computed {
    let shape = shape.into();
    let y = Computed::new((**x).reshape(shape.as_slice()));

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "reshape",
        move |xs, _ys, gys| {
            let gx = gys[0].reshape(xs[0].shape());
            vec![gx]
        },
    );

    y
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = reshape(&x, vec![3, 2]);
        dbg!(&*y);
        assert_eq!(y.shape(), &[3, 2]);

        let grads = gradients(&[y], &[x.clone()], false);
        assert_eq!(grads[0].shape(), &[2, 3]);
    }
}
