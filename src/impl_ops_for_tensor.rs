use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &Tensor {
            type Output = Tensor;

            fn $fn(self, rhs: Self) -> Self::Output {
                call!(functions::$op, self, rhs)
            }
        }

        impl std::ops::$op for Tensor {
            type Output = Tensor;

            fn $fn(self, rhs: Self) -> Self::Output {
                call!(functions::$op, self, rhs)
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
        call!(functions::Neg, self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        call!(functions::Neg, self)
    }
}

impl Tensor {
    pub fn broadcast(&self, shape: impl Into<Vec<usize>>) -> Tensor {
        call!(functions::Broadcast::new(shape.into()), self)
    }

    pub fn exp(&self) -> Tensor {
        call!(functions::Exp, self)
    }

    pub fn mat_t(&self) -> Tensor {
        call!(functions::MatTranspose, self)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        call!(functions::Matmul, self, rhs)
    }

    pub fn pow(&self, rhs: f32) -> Tensor {
        call!(functions::Pow::new(rhs), self)
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Tensor {
        call!(functions::Reshape::new(shape.into()), self)
    }

    pub fn sin(&self) -> Tensor {
        call!(functions::Sin, self)
    }

    pub fn cos(&self) -> Tensor {
        call!(functions::Cos, self)
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_arg: I,
    ) -> Tensor {
        call!(functions::Slice::new(slice_arg), self)
    }

    pub fn sum(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Tensor {
        call!(functions::Sum::new(axes.into(), keep_dim), self)
    }

    pub fn t(&self) -> Tensor {
        call!(functions::T, self)
    }

    pub fn tanh(&self) -> Tensor {
        call!(functions::Tanh, self)
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> Tensor {
        call!(functions::Transpose::new(axes.into()), self)
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
