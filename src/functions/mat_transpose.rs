use crate::*;

pub fn mat_transpose(x: &ComputedNDA) -> ComputedNDA {
    assert!(x.shape().len() >= 2);
    let y = ComputedNDA::new(forward(&**x));

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "mat_transpose",
        |_xs, _ys, gys| {
            let gx = gys[0].mat_t();
            vec![gx]
        },
    );

    y
}

pub fn forward(x: &NDArray) -> NDArray {
    let mut axes: Vec<_> = (0..x.shape().len()).collect();
    axes[x.shape().len() - 2..].reverse();

    x.view().permuted_axes(axes).into_ndarray()
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::Array::zeros([1, 2, 3]).into_ndarray());
        let y = mat_transpose(&x);
        assert_eq!(y.shape(), &[1, 3, 2]);

        let grads = gradients(&[y.clone()], &[x.clone()], false);
        assert_eq!(grads[0].shape(), &[1, 2, 3]);
    }
}
