use crate::*;

pub fn broadcast(x: &Computed, shape: impl Into<Vec<usize>>) -> Computed {
    let shape = shape.into();
    let y = Computed::new(
        (**x)
            .broadcast(shape.as_slice())
            .unwrap_or_else(|| panic!("illegal broadcast: {:?} to {:?}", x.shape(), shape))
            .into_ndarray(),
    );

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "broadcast",
        move |xs, _ys, gys| {
            let mut axes = Vec::new();
            let mut target = xs[0].shape().to_vec();
            for (axis, size) in shape.iter().enumerate() {
                if let Some(s) = target.first() {
                    if s == size {
                        target.remove(0);
                        continue;
                    }
                }
                axes.push(axis);
            }

            let gx = gys[0].sum(axes, false);

            vec![gx]
        },
    );

    y
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = broadcast(&x, vec![2, 3]);
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(&*y, &*x);
    }

    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = broadcast(&x, vec![4, 2, 3]);
        // dbg!(&*y);
        assert_eq!(y.shape(), &[4, 2, 3]);
    }
}
