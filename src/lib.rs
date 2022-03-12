mod variable;
mod function;
mod funcall;

pub use variable::*;
pub use function::*;
pub use funcall::*;

pub struct Double;

impl Function for Double {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);
        vec![Variable::new(*xs[0] * 2.0)]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![Variable::new(*gys[0] * 2.0)]
    }
}

pub struct Square;

impl Function for Square {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);
        vec![Variable::new(*xs[0] * *xs[0])]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![Variable::new(*gys[0] * 2.0 * *xs[0])]
    }
}

pub struct Exp;

impl Function for Exp {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);
        vec![Variable::new(xs[0].exp())]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![Variable::new(*gys[0] * xs[0].exp())]
    }
}

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
