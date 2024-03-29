use ndarray::Axis;
use ndarray::{ArrayBase, OwnedArcRepr};
use ndarray_rand::rand_distr::num_traits;

use crate::*;

type NDArray<T> = ArrayBase<OwnedArcRepr<T>, ndarray::IxDyn>;

pub fn sum<T: Clone + std::ops::Add<Output = T> + num_traits::Zero + Send + Sync + 'static>(
    x: &Computed<NDArray<T>>,
    axes: impl Into<Vec<usize>>,
    keep_dim: bool,
) -> Computed<NDArray<T>> {
    let axes = axes.into();
    let mut y = (**x).to_owned();
    for axis in axes.iter().rev() {
        y = y.sum_axis(Axis(*axis));
        if keep_dim {
            y.insert_axis_inplace(Axis(*axis));
        }
    }
    let y = Computed::new(y.into_shared().into_dyn());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "sum",
        move |xs, _ys, gys| {
            let gx = super::broadcast(&gys[0], xs[0].shape());

            vec![gx]
        },
    );

    y
}

pub fn sum_axes_to_desire(src_shape: &[usize], dst_shape: &[usize]) -> Vec<usize> {
    assert!(src_shape.len() >= dst_shape.len());
    let offset = src_shape.len() - dst_shape.len();
    let mut axes: Vec<_> = (0..offset).collect();
    for axis in offset..src_shape.len() {
        if dst_shape[axis - offset] == 1 {
            axes.push(axis);
        } else {
            assert!(src_shape[axis] == dst_shape[axis - offset]);
        }
    }
    axes
}

#[test]
fn test_sum_axes_to_desire() {
    assert_eq!(sum_axes_to_desire(&[2, 3, 4], &[2, 1, 4]), vec![1]);
}

#[test]
fn test() {
    {
        let x = ComputedNDA::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = sum(&x, vec![0], false);
        assert_eq!(y.shape(), &[3]);
        assert_eq!(&*y, &ndarray::array![5., 7., 9.].into_ndarray());
    }

    {
        let x = ComputedNDA::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let y = sum(&x, vec![1], false);
        assert_eq!(y.shape(), &[2]);
        assert_eq!(&*y, &ndarray::array![6., 15.].into_ndarray());
    }
}
