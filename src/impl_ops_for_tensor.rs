use crate::*;

macro_rules! impl_op {
    ($op:ident, $fn:ident) => {
        impl std::ops::$op for &ComputedNDA {
            type Output = ComputedNDA;

            fn $fn(self, rhs: Self) -> Self::Output {
                functions::$fn(self, rhs)
            }
        }

        impl std::ops::$op for ComputedNDA {
            type Output = ComputedNDA;

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

impl std::ops::Neg for &ComputedNDA {
    type Output = ComputedNDA;

    fn neg(self) -> Self::Output {
        functions::neg(self)
    }
}

impl std::ops::Neg for ComputedNDA {
    type Output = ComputedNDA;

    fn neg(self) -> Self::Output {
        functions::neg(&self)
    }
}

impl ComputedNDA {
    pub fn abs(&self) -> ComputedNDA {
        functions::abs(self)
    }

    pub fn broadcast(&self, shape: impl Into<Vec<usize>>) -> ComputedNDA {
        functions::broadcast(self, shape)
    }

    pub fn exp(&self) -> ComputedNDA {
        functions::exp(self)
    }

    pub fn log(&self) -> ComputedNDA {
        functions::log(self)
    }

    pub fn mat_t(&self) -> ComputedNDA {
        functions::mat_transpose(self)
    }

    pub fn matmul(&self, rhs: &ComputedNDA) -> ComputedNDA {
        functions::matmul(self, rhs)
    }

    pub fn pow(&self, rhs: f32) -> ComputedNDA {
        functions::pow(self, rhs)
    }

    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> ComputedNDA {
        functions::reshape(self, shape)
    }

    pub fn sin(&self) -> ComputedNDA {
        functions::sin(self)
    }

    pub fn cos(&self) -> ComputedNDA {
        functions::cos(self)
    }

    pub fn slice<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_arg: I,
    ) -> ComputedNDA {
        functions::slice(self, slice_arg)
    }

    pub fn slices<I: ndarray::SliceArg<ndarray::IxDyn> + Clone + Sync + Send + 'static>(
        &self,
        slice_args: Vec<I>,
    ) -> Vec<ComputedNDA> {
        functions::slices(self, slice_args)
    }

    pub fn sum(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> ComputedNDA {
        functions::sum(self, axes, keep_dim)
    }

    pub fn t(&self) -> ComputedNDA {
        functions::t(self)
    }

    pub fn tanh(&self) -> ComputedNDA {
        functions::tanh(self)
    }

    pub fn transpose(&self, axes: impl Into<Vec<usize>>) -> ComputedNDA {
        functions::transpose(self, axes)
    }
}

#[test]
fn test() {
    let x = ComputedNDA::new(scalar(1.0));
    let y = ComputedNDA::new(scalar(2.0));
    let z = x + y;
    assert_eq!(z[[]], 3.0);

    let x = ComputedNDA::new(scalar(1.0));
    let y = ComputedNDA::new(scalar(2.0));
    let z = &x - &y;
    assert_eq!(z[[]], -1.0);
}
