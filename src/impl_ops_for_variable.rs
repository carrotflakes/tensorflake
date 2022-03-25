use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &Tensor {
            type Output = Tensor;

            fn $fn(self, rhs: Self) -> Self::Output {
                call!(functions::$op, self.clone(), rhs.clone())
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
        call!(functions::Neg, self.clone())
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
        call!(functions::Broadcast::new(shape.into()), self.clone())
    }

    pub fn exp(&self) -> Tensor {
        call!(functions::Exp, self.clone())
    }

    pub fn mat_t(&self) -> Tensor {
        call!(functions::MatTranspose, self.clone())
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        call!(functions::Matmul, self.clone(), rhs.clone())
    }

    pub fn pow(&self, rhs: f32) -> Tensor {
        call!(functions::Pow::new(rhs), self.clone())
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Tensor {
        call!(functions::Reshape::new(shape.into()), self.clone())
    }

    pub fn sin(&self) -> Tensor {
        call!(functions::Sin, self.clone())
    }

    pub fn cos(&self) -> Tensor {
        call!(functions::Cos, self.clone())
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + 'static>(
        &self,
        slice_arg: I,
    ) -> Tensor {
        call!(functions::Slice::new(slice_arg), self.clone())
    }

    pub fn sum(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Tensor {
        call!(functions::Sum::new(axes.into(), keep_dim), self.clone())
    }

    pub fn t(&self) -> Tensor {
        call!(functions::T, self.clone())
    }

    pub fn tanh(&self) -> Tensor {
        call!(functions::Tanh, self.clone())
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> Tensor {
        call!(functions::Transpose::new(axes.into()), self.clone())
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
