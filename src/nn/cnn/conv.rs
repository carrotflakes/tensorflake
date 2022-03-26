use crate::{
    functions::*,
    nn::im2col::{get_conv_outsize, Im2col},
    *,
};

pub struct Conv2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub w: Param, // [out_ch, in_ch, kh, kw]
    pub b: Param, // [out_ch]
}

impl Conv2d {
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        w: Param,
        b: Param,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            w,
            b,
        }
    }
}

impl Layer for Conv2d {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output
    where
        Self: Sized + 'static,
    {
        let oh = get_conv_outsize(
            x.shape()[2],
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        );
        let ow = get_conv_outsize(
            x.shape()[3],
            self.kernel_size[1],
            self.stride[1],
            self.padding[1],
        );
        let col = Im2col::new(self.kernel_size, self.stride, self.padding)
            .call(vec![x.clone()])
            .pop()
            .unwrap();
        let w = self.w.get_tensor();
        let oc = w.shape()[0];
        let w = call!(
            T,
            call!(
                Reshape::new(vec![w.shape()[0], w.shape().iter().skip(1).product()]),
                w
            )
        );
        let b = self.b.get_tensor();
        let t = call!(Add, call!(Matmul, col, w), b);
        call!(
            Transpose::new(vec![0, 3, 1, 2]),
            call!(Reshape::new(vec![x.shape()[0], oh, ow, oc]), t)
        )
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

#[test]
fn test_conv2d() {
    use ndarray::prelude::*;
    let x = backprop(
        Array::from_shape_vec((1, 3, 4, 4), (0..16 * 3).map(|x| x as f32).collect())
            .unwrap()
            .into_ndarray(),
    );
    let w = Array::from_shape_vec((3, 3, 3, 3), (0..3usize.pow(4)).map(|x| x as f32).collect())
        .unwrap()
        .into_ndarray();
    let b = Array::from_shape_vec((3,), (0..3).map(|x| x as f32).collect())
        .unwrap()
        .into_ndarray();
    let conv = Conv2d {
        kernel_size: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        w: Fixed::new(w),
        b: Fixed::new(b),
    };
    let y = conv.call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 3, 4, 4]);
    dbg!(&*y);
    // export_dot::export_dot(&y, "conv2d.dot").unwrap();

    let grads = gradients(&[y], &[x.clone()], true);
    dbg!(&*grads[0]);
}
