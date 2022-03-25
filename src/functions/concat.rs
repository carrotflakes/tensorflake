use ndarray::{Axis, IxDyn, SliceInfo, SliceInfoElem};

use crate::functions::*;
use crate::*;

pub struct Concat {
    axis: usize,
}

impl Concat {
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl Function for Concat {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        let concated = ndarray::concatenate(
            Axis(self.axis),
            &xs.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        vec![concated.into_ndarray().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Tensor>,
        ys: &Vec<Tensor>,
        gys: &Vec<Tensor>,
    ) -> Vec<Tensor> {
        drop(ys);

        let gy = &gys[0];
        let mut acc = 0;
        xs.iter()
            .map(move |x| {
                let slice_info = SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::try_from(
                    (0..x.ndim())
                        .map(|i| {
                            if i == self.axis {
                                SliceInfoElem::Slice {
                                    start: acc as isize,
                                    end: Some((acc + x.shape()[self.axis]) as isize),
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
                let v = call!(Slice::new(slice_info), gy);
                acc += x.shape()[self.axis];
                v
            })
            .collect()
    }
}

#[test]
fn test() {
    use ndarray::{array, Array};
    let a = backprop(Array::zeros([2, 3]).into_ndarray());
    let b = backprop(Array::ones([2, 3]).into_ndarray());
    let y = call!(Concat::new(0), a, b);
    let y = call!(
        Mul,
        y,
        backprop(
            array![
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0]
            ]
            .into_ndarray()
        )
    );
    assert_eq!(y.shape(), [4, 3]);

    let gs = gradients(&[y], &[a.clone(), b.clone()], false);
    assert_eq!(gs[0].shape(), [2, 3]);
    assert_eq!(gs[1].shape(), [2, 3]);
    // dbg!(&*gs[0]);
    // dbg!(&*gs[1]);

    let y = call!(Concat::new(1), a, b);
    assert_eq!(y.shape(), [2, 6]);
}
