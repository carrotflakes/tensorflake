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
