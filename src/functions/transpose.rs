use crate::*;

pub fn transpose(x: &Computed, axes: impl Into<Vec<usize>>) -> Computed {
    let axes = axes.into();
    assert!((0..axes.len()).all(|i| axes.contains(&i)));

    let y = Computed::new(x.view().permuted_axes(&*axes).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "transpose",
        move |_xs, _ys, gys| {
            let gx = gys[0].transpose(
                (0..axes.len())
                    .map(|i| axes.iter().position(|j| *j == i).unwrap())
                    .collect::<Vec<_>>(),
            );
            vec![gx]
        },
    );

    y
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::Array::zeros([1, 2, 3]).into_ndarray());
        let y = transpose(&x, vec![1, 2, 0]);
        assert_eq!(y.shape(), &[2, 3, 1]);

        let grads = gradients(&[y], &[x], false);
        assert_eq!(grads[0].shape(), &[1, 2, 3]);
    }
}
