use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &Variable {
            type Output = Variable;

            fn $fn(self, rhs: Self) -> Self::Output {
                call!(functions::$op, self.clone(), rhs.clone())
            }
        }

        impl std::ops::$op for Variable {
            type Output = Variable;

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

impl std::ops::Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        call!(functions::Neg, self.clone())
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        call!(functions::Neg, self)
    }
}

impl Variable {
    pub fn broadcast_to(&self, shape: impl Into<Vec<usize>>) -> Variable {
        call!(functions::BroadcastTo::new(shape.into()), self.clone())
    }

    pub fn exp(&self) -> Variable {
        call!(functions::Exp, self.clone())
    }

    pub fn mat_t(&self) -> Variable {
        call!(functions::MatTranspose, self.clone())
    }

    pub fn matmul(&self, rhs: &Variable) -> Variable {
        call!(functions::Matmul, self.clone(), rhs.clone())
    }

    pub fn pow(&self, rhs: f32) -> Variable {
        call!(functions::Pow::new(rhs), self.clone())
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Variable {
        call!(functions::Reshape::new(shape.into()), self.clone())
    }

    pub fn sin(&self) -> Variable {
        call!(functions::Sin, self.clone())
    }

    pub fn cos(&self) -> Variable {
        call!(functions::Cos, self.clone())
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + 'static>(
        &self,
        slice_arg: I,
    ) -> Variable {
        call!(functions::Slice::new(slice_arg), self.clone())
    }

    pub fn sum_to(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Variable {
        call!(functions::SumTo::new(axes.into(), keep_dim), self.clone())
    }

    pub fn t(&self) -> Variable {
        call!(functions::T, self.clone())
    }

    pub fn tanh(&self) -> Variable {
        call!(functions::Tanh, self.clone())
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> Variable {
        call!(functions::Transpose::new(axes.into()), self.clone())
    }
}

#[test]
fn test() {
    let x = Variable::new(scalar(1.0));
    let y = Variable::new(scalar(2.0));
    let z = x + y;
    assert_eq!(z[[]], 3.0);

    let x = Variable::new(scalar(1.0));
    let y = Variable::new(scalar(2.0));
    let z = &x - &y;
    assert_eq!(z[[]], -1.0);
}
