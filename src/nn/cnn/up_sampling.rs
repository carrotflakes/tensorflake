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

impl Function for UpSampling2d {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert_eq!(xs.len(), 1);
        let x = &*xs[0];
        let s = x.shape();
        let [sh, sw] = self.size;
        let mut y = Array4::zeros([s[0], s[1], s[2] * sh, s[3] * sw]);

        for i in 0..sh {
            for j in 0..sw {
                y.slice_mut(s![.., .., i..;sh, j..;sw]).assign(x);
            }
        }

        vec![y.into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        drop(xs);
        drop(ys);
        let gy = naive_sum_pooling(&gys[0], self.size, self.size, [0, 0]);
        vec![gy]
    }
}

#[test]
fn test() {
    let x = backprop(
        Array4::from_shape_vec([1, 2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.])
            .unwrap()
            .into_tensor(),
    );
    let y = UpSampling2d::new([2, 2]).call(vec![x.clone()]);
    assert_eq!(y[0].shape(), &[1, 2, 4, 4]);
    dbg!(&*y[0]);

    let y = UpSampling2d::new([1, 3]).call(vec![x.clone()]);
    assert_eq!(y[0].shape(), &[1, 2, 2, 6]);

    let y = UpSampling2d::new([4, 2]).call(vec![x.clone()]);
    assert_eq!(y[0].shape(), &[1, 2, 8, 4]);
}
