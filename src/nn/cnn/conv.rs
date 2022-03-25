use crate::{
    functions::*,
    nn::im2col::{get_conv_outsize, Im2col},
    *,
};

pub struct Conv2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub w: Optimizee, // [out_ch, in_ch, kh, kw]
    pub b: Optimizee, // [out_ch]
}

impl Conv2d {
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        w: Optimizee,
        b: Optimizee,
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
    fn call(&self, xs: Vec<Tensor>, _train: bool) -> Vec<Tensor>
    where
        Self: Sized + 'static,
    {
        assert_eq!(xs.len(), 1);

        let oh = get_conv_outsize(
            xs[0].shape()[2],
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
        );
        let ow = get_conv_outsize(
            xs[0].shape()[3],
            self.kernel_size[1],
            self.stride[1],
            self.padding[1],
        );
        let col = Im2col::new(self.kernel_size, self.stride, self.padding)
            .call(xs.clone())
            .pop()
            .unwrap();
        let w = self.w.get();
        let oc = w.shape()[0];
        let w = call!(
            T,
            call!(
                Reshape::new(vec![w.shape()[0], w.shape().iter().skip(1).product()]),
                w
            )
        );
        let b = self.b.get();
        let t = call!(Add, call!(Matmul, col, w), b);
        vec![call!(
            Transpose::new(vec![0, 3, 1, 2]),
            call!(Reshape::new(vec![xs[0].shape()[0], oh, ow, oc]), t)
        )]
    }

    fn all_optimizees(&self) -> Vec<Optimizee> {
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
    let ys = conv.call(vec![x.clone()], false);
    assert_eq!(ys[0].shape(), &[1, 3, 4, 4]);
    dbg!(&*ys[0]);
    // export_dot::export_dot(&y, "conv2d.dot").unwrap();

    let grads = gradients(&ys, &[x.clone()], true);
    dbg!(&*grads[0]);
}
