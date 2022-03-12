use crate::{Function, Variable};

pub struct Double;

impl Function for Double {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);
        vec![Variable::new(*xs[0] * 2.0)]
    }

    fn backward(&self, _xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
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
