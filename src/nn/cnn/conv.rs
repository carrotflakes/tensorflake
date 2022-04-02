use ndarray::Axis;

use crate::{
    functions::*,
    nn::im2col::{get_conv_outsize, Im2col},
    *,
};

use super::im2col::{col2im, im2col};

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
        let t = matmul_add(&col, &w, &b);
        call!(
            Transpose::new(vec![0, 3, 1, 2]),
            call!(Reshape::new(vec![x.shape()[0], oh, ow, oc]), t)
        )

        // The implementation using tensordot, but it is slower than the implementation above.
        // conv2d(
        //     self.stride,
        //     self.padding,
        //     &self.w.get_tensor(),
        //     Some(&self.b.get_tensor()),
        //     &x,
        // )
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
        w: Param::new(w, optimizers::Fixed),
        b: Param::new(b, optimizers::Fixed),
    };
    let y = conv.call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 3, 4, 4]);
    dbg!(&*y);
    // export_dot::export_dot(&y, "conv2d.dot").unwrap();

    let grads = gradients(&[y], &[x.clone()], true);
    dbg!(&*grads[0]);
}

pub fn conv2d(
    stride: [usize; 2],
    padding: [usize; 2],
    kernel: &Tensor,
    bias: Option<&Tensor>,
    x: &Tensor,
) -> Tensor {
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let col = im2col(x, [kh, kw], stride, padding, false);

    let mut y = ndarray_util::tensordot(
        &col,
        kernel,
        &[Axis(1), Axis(2), Axis(3)],
        &[Axis(1), Axis(2), Axis(3)],
    );

    if let Some(bias) = bias {
        y += &**bias;
    }
    y = y.permuted_axes(&[0, 3, 1, 2][..]);

    let y = Tensor::new(y.into_ndarray());

    let mut xs = vec![x.clone(), kernel.clone()];
    xs.extend(bias.cloned());
    chain(&xs, &[y.clone()], false, "conv2d", move |xs, _, gys| {
        let gx = conv2d_transposed(
            stride,
            padding,
            [xs[0].shape()[2], xs[0].shape()[3]],
            &xs[1],
            None,
            &gys[0],
        );
        let gw = conv2d_grad_w(stride, padding, [kh, kw], &xs[0], &gys[0]);

        match xs.len() {
            2 => {
                vec![gx, gw]
            }
            3 => {
                let gb = gys[0].sum(vec![0, 2, 3], false);
                vec![gx, gw, gb]
            }
            _ => panic!(),
        }
    });

    y
}

pub fn conv2d_transposed(
    stride: [usize; 2],
    padding: [usize; 2],
    out_size: [usize; 2],
    kernel: &Tensor,
    bias: Option<&Tensor>,
    x: &Tensor,
) -> Tensor {
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let img_shape = [x.shape()[0], x.shape()[1], out_size[0], out_size[1]];

    let gcol = ndarray_util::tensordot(kernel, x, &[Axis(0)], &[Axis(1)]);
    let gcol = gcol.permuted_axes(&[3, 0, 1, 2, 4, 5][..]);
    let mut y = col2im(
        &gcol.into_ndarray(),
        img_shape,
        [kh, kw],
        stride,
        padding,
        false,
    );

    if let Some(bias) = bias {
        y += &(**bias).reshape([1, bias.len(), 1, 1]);
    }

    let y = Tensor::new(y);

    let mut xs = vec![x.clone(), kernel.clone()];
    xs.extend(bias.cloned());
    chain(
        &xs,
        &[y.clone()],
        false,
        "conv2d_transposed",
        move |xs, _, gys| {
            let gx = conv2d(stride, padding, &xs[1], None, &gys[0]);
            let gw = conv2d_grad_w(stride, padding, [kh, kw], &gys[0], &xs[0]);

            match xs.len() {
                2 => {
                    vec![gx, gw]
                }
                3 => {
                    let gb = gys[0].sum(vec![0, 2, 3], false);
                    vec![gx, gw, gb]
                }
                _ => panic!(),
            }
        },
    );

    y
}

pub fn conv2d_grad_w(
    stride: [usize; 2],
    padding: [usize; 2],
    kernel_size: [usize; 2],
    x: &Tensor,
    gy: &Tensor,
) -> Tensor {
    let col = im2col(x, kernel_size, stride, padding, false);

    let gw = ndarray_util::tensordot(
        gy,
        &col,
        &[Axis(0), Axis(2), Axis(3)],
        &[Axis(0), Axis(4), Axis(5)],
    );
    let gw = Tensor::new(gw.into_ndarray());

    chain(
        &[x.clone(), gy.clone()],
        &[gw.clone()],
        false,
        "conv2d_grad_w",
        move |xs, ys, _| {
            let gx = conv2d_transposed(
                stride,
                padding,
                [xs[0].shape()[2], xs[0].shape()[3]],
                &ys[0],
                None,
                &xs[1],
            );
            let ggy = conv2d(stride, padding, &ys[0], None, &xs[0]);
            vec![gx, ggy]
        },
    );

    gw
}
