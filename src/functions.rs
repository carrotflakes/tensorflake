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

pub struct Sum;

impl Function for Sum {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
        assert!(xs.len() >= 1);
        let mut y = *xs[0].clone();
        for x in xs.iter().skip(1) {
            y += **x;
        }
        vec![Variable::new(y)]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        (0..xs.len()).map(|_| gys[0].clone()).collect()
    }
}

#[test]
fn test_sum() {
    {
        let x = Variable::new(1.0);
        let y = Variable::new(2.0);
        let z = Variable::new(3.0);
        let xs = vec![x.clone(), y.clone(), z.clone()];
        let ys = Sum.call(xs);
        assert_eq!(*ys[0], 6.0);

        ys[0].set_grad(Variable::new(1.0));
        ys[0].backward();
        assert_eq!(*x.get_grad().unwrap(), 1.0);
        assert_eq!(*y.get_grad().unwrap(), 1.0);
        assert_eq!(*z.get_grad().unwrap(), 1.0);
    }
    {
        let x = Variable::new(3.0);
        Sum.call(vec![x.clone(), x.clone()]);
        let ys = Sum.call(vec![x.clone(), x.clone()]);
        assert_eq!(*ys[0], 6.0);

        ys[0].set_grad(Variable::new(1.0));
        ys[0].backward();
        assert_eq!(*x.get_grad().unwrap(), 2.0);
    }
}
