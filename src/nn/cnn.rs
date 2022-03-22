use std::ops::AddAssign;

use ndarray::{s, Array, Array4, Array6};

use crate::{functions::*, *};

pub struct Conv2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    w: Box<dyn Fn() -> Variable>, // [out_ch, in_ch, kh, kw]
    b: Box<dyn Fn() -> Variable>, // [out_ch]
}

impl Layer for Conv2d {
    fn call(&self, xs: Vec<Variable>, _train: bool) -> Vec<Variable>
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
        let w = (self.w)();
        let oc = w.shape()[0];
        let w = call!(
            T,
            call!(
                Reshape::new(vec![w.shape()[0], w.shape().iter().skip(1).product()]),
                w
            )
        );
        let b = (self.b)();
        let t = call!(Add, call!(Matmul, col, w), b);
        vec![call!(
            Transpose::new(vec![0, 3, 1, 2]),
            call!(Reshape::new(dbg!(vec![xs[0].shape()[0], oh, ow, oc])), t)
        )]
    }

    fn all_params(&self) -> Vec<Variable> {
        todo!()
    }
}

#[test]
fn test_conv2d() {
    let x = Variable::new(
        Array::from_shape_vec((1, 3, 4, 4), (0..16 * 3).map(|x| x as f32).collect())
            .unwrap()
            .into_tensor(),
    );
    let w = Variable::new(
        Array::from_shape_vec((3, 3, 3, 3), (0..3usize.pow(4)).map(|x| x as f32).collect())
            .unwrap()
            .into_tensor(),
    );
    let b = Variable::new(
        Array::from_shape_vec((3,), (0..3).map(|x| x as f32).collect())
            .unwrap()
            .into_tensor(),
    );
    let conv = Conv2d {
        kernel_size: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        w: Box::new(move || w.clone()),
        b: Box::new(move || b.clone()),
    };
    let y = conv.call(vec![x], false);
    assert_eq!(y[0].shape(), &[1, 3, 4, 4]);
    dbg!(&*y[0]);
}

pub struct Im2col {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl Im2col {
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Function for Im2col {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert_eq!(xs.len(), 1);
        vec![im2col(&*xs[0], self.kernel_size, self.stride, self.padding).into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        drop(ys);
        Col2im::new(
            xs[0].shape().try_into().unwrap(),
            self.kernel_size,
            self.stride,
            self.padding,
        )
        .call(gys.clone())
    }
}

pub struct Col2im {
    pub input_shape: [usize; 4],
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl Col2im {
    pub fn new(
        input_shape: [usize; 4],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Self {
        Self {
            input_shape,
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Function for Col2im {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert_eq!(xs.len(), 1);
        vec![im2col(&*xs[0], self.kernel_size, self.stride, self.padding).into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        drop(xs);
        drop(ys);
        Im2col::new(self.kernel_size, self.stride, self.padding).call(gys.clone())
    }
}

pub fn get_conv_outsize(input_size: usize, kernel_size: usize, stride: usize, pad: usize) -> usize {
    (input_size + pad * 2 - kernel_size) / stride + 1
}

pub fn im2col(
    x: &Tensor,
    [kh, kw]: [usize; 2],
    [sh, sw]: [usize; 2],
    [ph, pw]: [usize; 2],
) -> Tensor {
    assert_eq!(x.ndim(), 4);
    let s = x.shape();
    let oh = get_conv_outsize(s[2], kh, sh, ph);
    let ow = get_conv_outsize(s[3], kw, sw, pw);

    let f = |x: &Tensor| {
        let mut cols = Array6::zeros([s[0], s[1], kh, kw, oh, ow]);
        for h in 0..kh {
            let hlim = h + sh * oh;
            for w in 0..kw {
                let wlim = w + sw * ow;
                cols.slice_mut(s![.., .., h, w, .., ..]).assign(&x.slice(s![
                    ..,
                    ..,
                    h..hlim; sh,
                    w..wlim; sw,
                ]));
            }
        }
        cols
    };

    let cols = if ph == 0 && pw == 0 {
        f(x)
    } else {
        let mut y = Array::zeros([s[0], s[1], s[2] + ph * 2, s[3] + pw * 2]);
        y.slice_mut(s![.., .., ph..s[2] + ph, pw..s[3] + ph])
            .assign(&x);
        f(&y.into_tensor())
    };

    cols.permuted_axes([0, 4, 5, 1, 2, 3])
        .to_shape([s[0] * oh * ow, s[1] * kh * kw])
        .unwrap()
        .into_tensor()
}

#[test]
fn test_im2col() {
    use ndarray::prelude::*;
    let x = array![
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    ]
    .insert_axis(Axis(0))
    .into_tensor();
    let cols = im2col(&x, [2, 2], [2, 2], [1, 1]);
    dbg!(&cols);
}

pub fn col2im(
    x: &Tensor,
    img_shape: [usize; 4],
    [kh, kw]: [usize; 2],
    [sh, sw]: [usize; 2],
    [ph, pw]: [usize; 2],
) -> Tensor {
    assert_eq!(x.ndim(), 2);
    let s = img_shape;
    let oh = get_conv_outsize(s[2], kh, sh, ph);
    let ow = get_conv_outsize(s[3], kw, sw, pw);

    let col = x
        .to_shape([s[0], oh, ow, s[1], kh, kw])
        .unwrap()
        .permuted_axes([0, 3, 4, 5, 1, 2]);

    let mut img = Array4::zeros([s[0], s[1], s[2] + 2 * ph + sh - 1, s[3] + 2 * pw + sw - 1]);
    for h in 0..kh {
        let hlim = h + sh * oh;
        for w in 0..kw {
            let wlim = w + sw * ow;
            img.slice_mut(s![
                ..,
                ..,
                h..hlim; sh,
                w..wlim; sw,
            ])
            .add_assign(&col.slice(s![.., .., h, w, .., ..]));
        }
    }

    img.slice(s![.., .., ph..s[2] + ph, pw..s[3] + pw])
        .into_tensor()
}

#[test]
fn test_col2im() {
    use ndarray::prelude::*;
    let x = array![
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    ]
    .insert_axis(Axis(0))
    .into_tensor();
    let cols = im2col(&x, [2, 2], [2, 2], [1, 1]);

    let img = col2im(&cols, x.shape().try_into().unwrap(), [2, 2], [2, 2], [1, 1]);

    assert_eq!(img, x);
}

// pub fn im2col(x: &Tensor, (fh, fw): (usize, usize)) -> Tensor {
//     assert_eq!(x.ndim(), 4);
//     let s = x.shape();
//     let oh = s[2] - fh + 1;
//     let ow = s[3] - fw + 1;
//     let mut cols = Array6::zeros([s[0], s[1], fh, fw, oh, ow]);
//     for h in 0..fh {
//         for w in 0..fw {
//             cols.slice_mut(s![.., .., h, w, .., ..]).assign(&x.slice(s![
//                 ..,
//                 ..,
//                 h..h + oh,
//                 w..w + ow
//             ]));
//         }
//     }
//     cols.permuted_axes([1, 2, 3, 0, 4, 5])
//         .to_shape([s[1] * fh * fw, s[0] * oh * ow])
//         .unwrap()
//         .into_tensor()
// }

// #[test]
// fn test_im2col() {
//     use ndarray::prelude::*;
//     let x = array![
//         [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
//         [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
//     ]
//     .insert_axis(Axis(0))
//     .into_tensor();
//     let cols = im2col(&x, (2, 2));
//     dbg!(&cols);
// }
