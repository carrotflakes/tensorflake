use ndarray::{Axis, IxDyn, SliceInfo, SliceInfoElem};

use crate::functions::*;
use crate::*;

pub fn concat(xs: &[ComputedNDA], axis: usize) -> ComputedNDA {
    let y =
        ndarray::concatenate(Axis(axis), &xs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
    let y = ComputedNDA::new(y.into_ndarray());

    chain(xs, &[y.clone()], false, "concat", move |xs, _ys, gys| {
        let gy = &gys[0];
        let mut acc = 0;
        xs.iter()
            .map(move |x| {
                let slice_info = SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::try_from(
                    (0..x.ndim())
                        .map(|i| {
                            if i == axis {
                                SliceInfoElem::Slice {
                                    start: acc as isize,
                                    end: Some((acc + x.shape()[axis]) as isize),
                                    step: 1,
                                }
                            } else {
                                SliceInfoElem::Slice {
                                    start: 0,
                                    end: None,
                                    step: 1,
                                }
                            }
                        })
                        .collect::<Vec<SliceInfoElem>>(),
                )
                .unwrap();
                let v = slice(&gy, slice_info);
                acc += x.shape()[axis];
                v
            })
            .collect()
    });

    y
}

#[test]
fn test() {
    use ndarray::{array, Array};
    let a = backprop(Array::zeros([2, 3]).into_ndarray());
    let b = backprop(Array::ones([2, 3]).into_ndarray());
    let y = concat(&[a.clone(), b.clone()], 0);
    let y = mul(
        &y,
        &backprop(
            array![
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0]
            ]
            .into_ndarray(),
        ),
    );
    assert_eq!(y.shape(), [4, 3]);

    let gs = gradients(&[y], &[a.clone(), b.clone()], false);
    assert_eq!(gs[0].shape(), [2, 3]);
    assert_eq!(gs[1].shape(), [2, 3]);
    // dbg!(&*gs[0]);
    // dbg!(&*gs[1]);

    let y = concat(&[a, b], 1);
    assert_eq!(y.shape(), [2, 6]);
}
