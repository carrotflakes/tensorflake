mod funcall;
mod function;
pub mod functions;
mod variable;

pub use funcall::*;
pub use function::*;
pub use variable::*;

// pub fn numerical_diff(f: impl Function + Sized + 'static, x: &Variable) -> Variable {
//     let eps = 1e-4;
//     let y1 = Box::new(f).call(vec![Variable::new(**x - eps)]);
//     let y2 = Box::new(f).call(vec![Variable::new(**x + eps)]);
//     Variable::new((*y2[0] - *y1[0]) / (2.0 * eps))
// }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
