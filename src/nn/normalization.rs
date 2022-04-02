use ndarray::Axis;

use crate::{ndarray_util::map_axes_keep_dim, *};

// TODO: infer time

pub struct Normalization {
    pub axes: Vec<usize>,
    pub gamma: Param,
    pub beta: Param,
    pub eps: f32, // 0.001
}

impl Normalization {
    pub fn new(axes: Vec<usize>, eps: f32, optimizer: impl Optimizer) -> Self {
        Self {
            axes,
            gamma: Param::new(scalar(1.0), optimizer.clone()),
            beta: Param::new(scalar(0.0), optimizer.clone()),
            eps,
        }
    }
}

impl Layer for Normalization {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        let mean = map_axes_keep_dim(&*x, &self.axes, |x| x.mean_axis(Axis(1)).unwrap());
        let var = map_axes_keep_dim(&*x, &self.axes, |x| x.var_axis(Axis(1), 1.0));
        (x - Tensor::new(mean.into_ndarray()))
            * (self.gamma.get_tensor()
                / Tensor::new((var + self.eps).map(|x| x.sqrt()).into_ndarray()))
            + self.beta.get_tensor()
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[test]
fn test() {
    let x = Tensor::new(ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_ndarray());
    let bn = Normalization::new(vec![0], 0.001, optimizers::AdamOptimizer::new());
    let y = bn.call(x, false);
    assert!((y.mean().unwrap() - 0.0).abs() < 1e-6);
    assert!((y.var(1.0) - 1.0).abs() < 0.01);

    let x = backprop(
        ndarray::Array::from_shape_vec(
            [2, 3, 4, 5],
            (0..2 * 3 * 4 * 5).map(|x| x as f32).collect(),
        )
        .unwrap()
        .into_ndarray(),
    );
    let bn = Normalization::new(vec![1, 2], 0.001, optimizers::AdamOptimizer::new());
    let y = bn.call(x.clone(), false);
    dbg!(&*y);
    assert_eq!(x.shape(), y.shape());

    let grads = gradients(&[y], &[x], false);
    dbg!(&*grads[0]);
}
