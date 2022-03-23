use crate::{
    functions::*,
    nn::im2col::{get_conv_outsize, Im2col},
    *,
};

pub struct Conv2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub w: Box<dyn Fn() -> Variable>, // [out_ch, in_ch, kh, kw]
    pub b: Box<dyn Fn() -> Variable>, // [out_ch]
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
    use ndarray::prelude::*;
    let x = backprop(
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
    let ys = conv.call(vec![x.clone()], false);
    assert_eq!(ys[0].shape(), &[1, 3, 4, 4]);
    dbg!(&*ys[0]);
    // export_dot::export_dot(&y, "conv2d.dot").unwrap();

    let grads = gradients(&ys, &[x.clone()], true);
    dbg!(&*grads[0]);
}
