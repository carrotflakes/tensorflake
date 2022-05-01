use ndarray::Axis;

use crate::{
    functions::*,
    initializers::Initializer,
    nn::im2col::{get_conv_outsize, Im2col},
    *,
};

use super::im2col::{col2im, get_transposed_conv_outsize, im2col, Col2im};

pub struct Conv2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub w: Param,         // [out_ch, in_ch, kh, kw]
    pub b: Option<Param>, // [out_ch]
}

impl Conv2d {
    pub fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        w: impl Initializer,
        b: Option<impl Initializer>,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            w: w.scope("w").initialize(&[
                output_channel,
                input_channel,
                kernel_size[0],
                kernel_size[1],
            ]),
            b: b.map(|b| b.scope("b").initialize(&[output_channel])),
        }
    }
}

impl Layer for Conv2d {
    type Input = Computed;
    type Output = Computed;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output
    where
        Self: Sized + 'static,
    {
        assert_eq!(x.ndim(), 4);
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
        let col = Im2col::new(self.kernel_size, self.stride, self.padding, true)
            .call(x.clone(), train);
        // col: [batch_size * oh * ow, in_ch * kh * kw]
        let w = self.w.get();
        let oc = w.shape()[0];
        let kernel = w
            .reshape(vec![w.shape()[0], w.shape().iter().skip(1).product()])
            .t();
        // w: [in_ch * kh * kw, out_ch]
        let t = if let Some(b) = &self.b {
            matmul_add(&col, &kernel, &b.get())
        } else {
            col.matmul(&kernel)
        };
        // t: [batch_size * oh * ow, out_ch]
        let y = t
            .reshape(vec![x.shape()[0], oh, ow, oc])
            .transpose(vec![0, 3, 1, 2]);

        // let stride = self.stride;
        // let padding = self.padding;
        // let kernel_size = self.kernel_size;
        // chain(
        //     &[x, w, b],
        //     &[y.clone()],
        //     false,
        //     "Conv2d",
        //     move |xs, _, gys| {
        //         let gx = Conv2dTranspose::new(
        //             stride,
        //             padding,
        //             [xs[0].shape()[2], xs[0].shape()[3]],
        //             Param::new((*xs[1]).clone(), optimizers::Fixed),
        //             None,
        //         )
        //         .call(gys[0].clone(), false);
        //         let gw = conv2d_grad_w(stride, padding, kernel_size, &xs[0], &gys[0]);

        //         match xs.len() {
        //             2 => {
        //                 vec![gx, gw]
        //             }
        //             3 => {
        //                 let gb = gys[0].sum(vec![0, 2, 3], false);
        //                 vec![gx, gw, gb]
        //             }
        //             _ => panic!(),
        //         }
        //     },
        // );

        y

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
        [self.w.clone()].into_iter().chain(self.b.clone()).collect()
    }
}

#[test]
fn test_conv2d() {
    use ndarray::prelude::*;
    let x = backprop(
        Array::from_shape_vec((1, 2, 3, 4), (0..24).map(|x| x as f32).collect())
            .unwrap()
            .into_ndarray(),
    );
    let w = Array::from_shape_vec((5, 2, 3, 3), (0..5 * 2 * 3 * 3).map(|x| x as f32).collect())
        .unwrap()
        .into_ndarray();
    let b = Array::from_shape_vec((5,), (0..5).map(|x| x as f32).collect())
        .unwrap()
        .into_ndarray();
    let conv = Conv2d {
        kernel_size: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        w: Param::new(w.clone(), "w".into(), optimizers::Fixed),
        b: Some(Param::new(b.clone(), "w".into(), optimizers::Fixed)),
    };
    let y = conv.call(x.clone(), false);
    assert_eq!(y.shape(), &[1, 5, 3, 4]);
    dbg!(&*y);
    // export_dot::export_dot(&y, "conv2d.dot").unwrap();

    let grads = gradients(&[y.clone()], &[x.clone()], true);
    // dbg!(&*grads[0]);

    let y2 = conv2d([1, 1], [1, 1], &Computed::new(w), Some(&Computed::new(b)), &x);
    assert_eq!(&*y, &*y2);

    let grads2 = gradients(&[y2.clone()], &[x.clone()], true);
    assert_eq!(&*grads[0], &*grads2[0]);
}

pub struct Conv2dTranspose {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub out_size: Option<[usize; 2]>,
    pub w: Param,         // [out_ch, in_ch, kh, kw]
    pub b: Option<Param>, // [out_ch]
}

impl Conv2dTranspose {
    pub fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        out_size: Option<[usize; 2]>,
        w: impl Initializer,
        b: Option<impl Initializer>,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            out_size,
            w: w.scope("w").initialize(&[
                output_channel,
                input_channel,
                kernel_size[0],
                kernel_size[1],
            ]),
            b: b.map(|b| b.scope("b").initialize(&[input_channel])),
        }
    }
}

impl Layer for Conv2dTranspose {
    type Input = Computed;
    type Output = Computed;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output
    where
        Self: Sized + 'static,
    {
        let kernel = self.w.get(); // [out_ch, in_ch, kh, kw]

        let [oh, ow] = if let Some(out_size) = &self.out_size {
            out_size.clone()
        } else {
            [
                get_transposed_conv_outsize(
                    x.shape()[2],
                    self.kernel_size[0],
                    self.stride[0],
                    self.padding[0],
                    0,
                ),
                get_transposed_conv_outsize(
                    x.shape()[3],
                    self.kernel_size[1],
                    self.stride[1],
                    self.padding[1],
                    0,
                ),
            ]
        };

        let img_shape = [x.shape()[0], kernel.shape()[1], oh, ow];

        let kernel_size = [kernel.shape()[2], kernel.shape()[3]];
        let kernel = kernel.reshape(vec![
            kernel.shape()[0],
            kernel.shape().iter().skip(1).product(),
        ]);
        // kernel: [out_ch, in_ch*kh*kw]

        // x: [batch, out_ch, oh, ow]
        let col = x.transpose(vec![0, 2, 3, 1]);
        let col = col.reshape(vec![col.shape().iter().take(3).product(), col.shape()[3]]);
        // col: [batch*oh*ow, out_ch]

        let col = col.matmul(&kernel);
        // col: [batch*oh*ow, in_ch*kh*kw]

        // col: batch_size, oh, ow, in_ch, kh, kw
        let mut y = Col2im::new(img_shape, kernel_size, self.stride, self.padding, true)
            .call(col.clone(), train);

        if let Some(b) = &self.b {
            let b = b.get();
            let b = b.reshape(vec![1, b.len(), 1, 1]);
            y = y + b;
        }

        y

        // if let Some(b) = &self.b {
        //     conv2d_transpose(
        //         self.stride,
        //         self.padding,
        //         self.out_size,
        //         &self.w.get_tensor(),
        //         Some(&b.get_tensor()),
        //         &x,
        //     )
        // } else {
        //     conv2d_transpose(
        //         self.stride,
        //         self.padding,
        //         self.out_size,
        //         &self.w.get_tensor(),
        //         None,
        //         &x,
        //     )
        // }
    }

    fn all_params(&self) -> Vec<Param> {
        [self.w.clone()].into_iter().chain(self.b.clone()).collect()
    }
}

// TODO: test that Conv2dTranspose is the same as conv2d_transpose

pub fn conv2d(
    stride: [usize; 2],
    padding: [usize; 2],
    kernel: &Computed,
    bias: Option<&Computed>,
    x: &Computed,
) -> Computed {
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

    let y = Computed::new(y.into_ndarray());

    let mut xs = vec![x.clone(), kernel.clone()];
    xs.extend(bias.cloned());
    chain(&xs, &[y.clone()], false, "conv2d", move |xs, _, gys| {
        let gx = conv2d_transpose(
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

pub fn conv2d_transpose(
    stride: [usize; 2],
    padding: [usize; 2],
    out_size: [usize; 2],
    kernel: &Computed, // [out_ch, in_ch, kh, kw]
    bias: Option<&Computed>,
    x: &Computed, // [batch, out_ch, oh, ow]
) -> Computed {
    let kh = kernel.shape()[2];
    let kw = kernel.shape()[3];

    let img_shape = [x.shape()[0], kernel.shape()[1], out_size[0], out_size[1]];

    let gcol = ndarray_util::tensordot(kernel, x, &[Axis(0)], &[Axis(1)]);
    // gcol: [in_ch, kh, kw, batch_size, oh, ow]
    let gcol = gcol.permuted_axes(&[3, 0, 1, 2, 4, 5][..]);

    // gcol: [batch_size, in_ch, kh, kw, oh, ow]
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

    let y = Computed::new(y);

    let mut xs = vec![x.clone(), kernel.clone()];
    xs.extend(bias.cloned());
    chain(
        &xs,
        &[y.clone()],
        false,
        "conv2d_transpose",
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
    x: &Computed,
    gy: &Computed,
) -> Computed {
    let col = im2col(x, kernel_size, stride, padding, false);

    let gw = ndarray_util::tensordot(
        gy,
        &col,
        &[Axis(0), Axis(2), Axis(3)],
        &[Axis(0), Axis(4), Axis(5)],
    );
    let gw = Computed::new(gw.into_ndarray());

    chain(
        &[x.clone(), gy.clone()],
        &[gw.clone()],
        false,
        "conv2d_grad_w",
        move |xs, ys, _| {
            let gx = conv2d_transpose(
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
