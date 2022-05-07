use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &Computed {
            type Output = Computed;

            fn $fn(self, rhs: Self) -> Self::Output {
                functions::$fn(self, rhs)
            }
        }

        impl std::ops::$op for Computed {
            type Output = Computed;

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

impl std::ops::Neg for &Computed {
    type Output = Computed;

    fn neg(self) -> Self::Output {
        functions::neg(self)
    }
}

impl std::ops::Neg for Computed {
    type Output = Computed;

    fn neg(self) -> Self::Output {
        functions::neg(&self)
    }
}

impl Computed {
    pub fn abs(&self) -> Computed {
        functions::abs(self)
    }

    pub fn broadcast(&self, shape: impl Into<Vec<usize>>) -> Computed {
        functions::broadcast(self, shape)
    }

    pub fn exp(&self) -> Computed {
        functions::exp(self)
    }

    pub fn log(&self) -> Computed {
        functions::log(self)
    }

    pub fn mat_t(&self) -> Computed {
        functions::mat_transpose(self)
    }

    pub fn matmul(&self, rhs: &Computed) -> Computed {
        functions::matmul(self, rhs)
    }

    pub fn pow(&self, rhs: f32) -> Computed {
        functions::pow(self, rhs)
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Computed {
        functions::reshape(self, shape)
    }

    pub fn sin(&self) -> Computed {
        functions::sin(self)
    }

    pub fn cos(&self) -> Computed {
        functions::cos(self)
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_arg: I,
    ) -> Computed {
        functions::slice(self, slice_arg)
    }

    pub fn slices<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_args: Vec<I>,
    ) -> Vec<Computed> {
        functions::slices(self, slice_args)
    }

    pub fn sum(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Computed {
        functions::sum(self, axes, keep_dim)
    }

    pub fn t(&self) -> Computed {
        functions::t(self)
    }

    pub fn tanh(&self) -> Computed {
        functions::tanh(self)
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> Computed {
        functions::transpose(self, axes)
    }
}

#[test]
fn test() {
    let x = Computed::new(scalar(1.0));
    let y = Computed::new(scalar(2.0));
    let z = x + y;
    assert_eq!(z[[]], 3.0);

    let x = Computed::new(scalar(1.0));
    let y = Computed::new(scalar(2.0));
    let z = &x - &y;
    assert_eq!(z[[]], -1.0);
}
