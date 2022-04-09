use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &Tensor {
            type Output = Tensor;

            fn $fn(self, rhs: Self) -> Self::Output {
                functions::$fn(self, rhs)
            }
        }

        impl std::ops::$op for Tensor {
            type Output = Tensor;

            fn $fn(self, rhs: Self) -> Self::Output {
                functions::$fn(&self, &rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        functions::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        functions::neg(&self)
    }
}

impl Tensor {
    pub fn broadcast(&self, shape: impl Into<Vec<usize>>) -> Tensor {
        functions::broadcast(self, shape)
    }

    pub fn exp(&self) -> Tensor {
        functions::exp(self)
    }

    pub fn mat_t(&self) -> Tensor {
        functions::mat_transpose(self)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        functions::matmul::matmul(self, rhs)
    }

    pub fn pow(&self, rhs: f32) -> Tensor {
        functions::pow(self, rhs)
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Tensor {
        functions::reshape(self, shape)
    }

    pub fn sin(&self) -> Tensor {
        functions::sin(self)
    }

    pub fn cos(&self) -> Tensor {
        functions::cos(self)
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_arg: I,
    ) -> Tensor {
        call!(functions::Slice::new(slice_arg), self)
    }

    pub fn slices<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_args: Vec<I>,
    ) -> Vec<Tensor> {
        functions::Slices::new(slice_args).call(vec![self.clone()])
    }

    pub fn sum(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Tensor {
        functions::sum(self, axes, keep_dim)
    }

    pub fn t(&self) -> Tensor {
        functions::t(self)
    }

    pub fn tanh(&self) -> Tensor {
        functions::tanh(self)
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> Tensor {
        functions::transpose(self, axes)
    }
}

#[test]
fn test() {
    let x = Tensor::new(scalar(1.0));
    let y = Tensor::new(scalar(2.0));
    let z = x + y;
    assert_eq!(z[[]], 3.0);

    let x = Tensor::new(scalar(1.0));
    let y = Tensor::new(scalar(2.0));
    let z = &x - &y;
    assert_eq!(z[[]], -1.0);
}
