use std::ops::AddAssign;

use ndarray::{s, Array, Array4, Array6};

use crate::*;

pub struct Im2col {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub to_matrix: bool,
}

impl Im2col {
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        to_matrix: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            to_matrix,
        }
    }
}

impl Function for Im2col {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert_eq!(xs.len(), 1);
        vec![im2col(
            &*xs[0],
            self.kernel_size,
            self.stride,
            self.padding,
            self.to_matrix,
        )
        .into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        drop(ys);
        Col2im::new(
            xs[0].shape().try_into().unwrap(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.to_matrix,
        )
        .call(gys.clone())
    }
}

pub struct Col2im {
    pub input_shape: [usize; 4],
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub to_matrix: bool,
}

impl Col2im {
    pub fn new(
        input_shape: [usize; 4],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        to_matrix: bool,
    ) -> Self {
        Self {
            input_shape,
            kernel_size,
            stride,
            padding,
            to_matrix,
        }
    }
}

impl Function for Col2im {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert_eq!(xs.len(), 1);
        vec![col2im(
            &*xs[0],
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
            self.to_matrix,
        )
        .into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        drop(xs);
        drop(ys);
        Im2col::new(self.kernel_size, self.stride, self.padding, self.to_matrix).call(gys.clone())
    }
}

pub fn im2col(
    x: &NDArray,
    [kh, kw]: [usize; 2],
    [sh, sw]: [usize; 2],
    [ph, pw]: [usize; 2],
    to_matrix: bool,
) -> NDArray {
    assert_eq!(x.ndim(), 4);
    let s = x.shape();
    let oh = get_conv_outsize(s[2], kh, sh, ph);
    let ow = get_conv_outsize(s[3], kw, sw, pw);

    let f = |x: &NDArray| {
        let mut cols = Array6::zeros([s[0], s[1], kh, kw, oh, ow]);
        for h in 0..kh {
            let hlim = h + sh * (oh - 1) + 1;
            for w in 0..kw {
                let wlim = w + sw * (ow - 1) + 1;
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
        y.slice_mut(s![.., .., ph..s[2] + ph, pw..s[3] + pw])
            .assign(&x);
        f(&y.into_ndarray())
    };

    if to_matrix {
        cols.permuted_axes([0, 4, 5, 1, 2, 3])
            .to_shape([s[0] * oh * ow, s[1] * kh * kw])
            .unwrap()
            .into_ndarray()
    } else {
        cols.into_ndarray()
    }
}

#[test]
fn test_im2col() {
    use ndarray::prelude::*;
    let x = array![
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    ]
    .insert_axis(Axis(0))
    .into_ndarray();
    let cols = im2col(&x, [2, 2], [2, 2], [1, 1], true);
    dbg!(&cols);
    assert_eq!(cols.shape(), [4, 8]);

    let cols = im2col(&x, [1, 2], [1, 2], [0, 0], true);
    dbg!(&cols);
    assert_eq!(cols.shape(), [3, 4]);

    let cols = im2col(&x, [1, 2], [2, 1], [0, 0], true);
    dbg!(&cols);
    assert_eq!(cols.shape(), [4, 4]);
}

pub fn col2im(
    x: &NDArray,
    img_shape: [usize; 4],
    [kh, kw]: [usize; 2],
    [sh, sw]: [usize; 2],
    [ph, pw]: [usize; 2],
    to_matrix: bool,
) -> NDArray {
    let s = img_shape;
    let oh = get_conv_outsize(s[2], kh, sh, ph);
    let ow = get_conv_outsize(s[3], kw, sw, pw);

    let col = if to_matrix {
        assert_eq!(x.ndim(), 2);
        x.reshape([s[0], oh, ow, s[1], kh, kw])
            .permuted_axes([0, 3, 4, 5, 1, 2])
            .into_dyn()
    } else {
        x.clone()
    };

    let mut img = Array4::zeros([s[0], s[1], s[2] + 2 * ph + sh - 1, s[3] + 2 * pw + sw - 1]);
    for h in 0..kh {
        let hlim = h + sh * (oh - 1) + 1;
        for w in 0..kw {
            let wlim = w + sw * (ow - 1) + 1;
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
        .into_ndarray()
}

#[test]
fn test_im2col_col2im() {
    use ndarray::prelude::*;
    let x = array![
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
    ]
    .insert_axis(Axis(0))
    .into_ndarray();
    let cols = im2col(&x, [2, 2], [2, 2], [1, 1], true);

    let img = col2im(
        &cols,
        x.shape().try_into().unwrap(),
        [2, 2],
        [2, 2],
        [1, 1],
        true,
    );

    assert_eq!(&img, &x);

    let cols = im2col(&x, [2, 2], [2, 2], [1, 1], false);

    let img = col2im(
        &cols,
        x.shape().try_into().unwrap(),
        [2, 2],
        [2, 2],
        [1, 1],
        false,
    );

    assert_eq!(&img, &x);

    let cols = im2col(&x, [1, 2], [2, 1], [0, 0], true);
    let img = col2im(
        &cols,
        x.shape().try_into().unwrap(),
        [1, 2],
        [2, 1],
        [0, 0],
        true,
    );
    dbg!(&img);
}

pub fn get_conv_outsize(input_size: usize, kernel_size: usize, stride: usize, pad: usize) -> usize {
    (input_size + pad * 2 - kernel_size) / stride + 1
}

#[test]
fn test_get_conv_outsize() {
    assert_eq!(get_conv_outsize(3, 1, 2, 0), 2);
    assert_eq!(get_conv_outsize(3, 2, 1, 0), 2);
    assert_eq!(get_conv_outsize(4, 1, 2, 0), 2);
    assert_eq!(get_conv_outsize(4, 2, 1, 0), 3);
    assert_eq!(get_conv_outsize(3, 1, 2, 1), 3);
    assert_eq!(get_conv_outsize(3, 2, 1, 1), 4);
}

pub fn get_transposed_conv_outsize(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> usize {
    stride * (input_size - 1) + kernel_size - 2 * pad + out_pad
}
