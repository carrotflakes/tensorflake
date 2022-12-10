use ndarray::{s, Array4};

use crate::*;

use super::naive_sum_pooling;

pub struct UpSampling2d {
    pub size: [usize; 2],
}

impl UpSampling2d {
    pub fn new(size: [usize; 2]) -> Self {
        Self { size }
    }
}

impl Layer for UpSampling2d {
    type Input = ComputedNDA;
    type Output = ComputedNDA;

    fn call(&self, input: Self::Input, _train: bool) -> Self::Output {
        let s = input.shape();
        let [sh, sw] = self.size;
        let mut y = Array4::zeros([s[0], s[1], s[2] * sh, s[3] * sw]);

        for i in 0..sh {
            for j in 0..sw {
                y.slice_mut(s![.., .., i..;sh, j..;sw]).assign(&input);
            }
        }

        let y = ComputedNDA::new(y.into_ndarray());

        let size = self.size;

        chain(
            &[input.clone()],
            &[y.clone()],
            false,
            "UpSampling2d",
            move |_xs, _ys, gys| vec![naive_sum_pooling(&gys[0], size, size, [0, 0])],
        );

        y
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        Vec::new()
    }
}

#[test]
fn test() {
    let x = backprop(
        Array4::from_shape_vec([1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.])
            .unwrap()
            .into_ndarray(),
    );
    let y = UpSampling2d::new([2, 2]).call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 2, 4, 4]);
    dbg!(&*y);

    let y = UpSampling2d::new([1, 3]).call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 2, 2, 6]);

    let y = UpSampling2d::new([4, 2]).call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 2, 8, 4]);
}
