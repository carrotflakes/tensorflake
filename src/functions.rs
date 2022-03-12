use crate::{Function, Tensor, Variable};

pub struct Square;

impl Function for Square {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].multiply(&xs[0])]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![Variable::new(
            gys[0].multiply(&xs[0]).multiply_with_scalar(2.0),
        )]
    }
}

pub struct Exp;

impl Function for Exp {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| x.exp())]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![Variable::new(gys[0].multiply(&xs[0].map(|x| x.exp())))]
    }
}

pub struct Sum;

impl Function for Sum {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut y = xs[0].inner.data.clone();
        for x in xs.iter().skip(1) {
            y = &y + &x.inner.data;
        }
        vec![y]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        (0..xs.len()).map(|_| gys[0].clone()).collect()
    }
}

pub struct Mul;

impl Function for Mul {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut data = xs[0].inner.data.data.clone();
        for x in xs.iter().skip(1) {
            for (a, b) in data.iter_mut().zip(&x.inner.data.data) {
                *a *= *b;
            }
        }
        vec![Tensor::new(data, &xs[0].inner.data.shape)]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        (0..xs.len())
            .map(|i| {
                let mut data = gys[0].inner.data.data.clone();
                for j in 0..xs.len() {
                    if j != i {
                        for (a, b) in data.iter_mut().zip(&xs[j].inner.data.data) {
                            *a *= *b;
                        }
                    }
                }
                Variable::new(Tensor::new(data, &xs[0].inner.data.shape))
            })
            .collect()
    }
}

pub struct Sub;

impl Function for Sub {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 2);
        assert_eq!(xs[0].inner.data.shape, xs[1].inner.data.shape);

        vec![Tensor::new(
            xs[0]
                .inner
                .data
                .data
                .iter()
                .zip(&xs[1].inner.data.data)
                .map(|(a, b)| a - b)
                .collect(),
            &xs[0].inner.data.shape,
        )]
    }

    fn backward(&self, _xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![
            gys[0].clone(),
            Variable::new(gys[0].multiply_with_scalar(-1.0)),
        ]
    }
}

#[test]
fn test_sum() {
    {
        let x = Variable::new(1.0.into());
        let y = Variable::new(2.0.into());
        let z = Variable::new(3.0.into());
        let xs = vec![x.clone(), y.clone(), z.clone()];
        let ys = Sum.call(xs);
        assert_eq!(*ys[0], 6.0.into());

        ys[0].set_grad(Variable::new(1.0.into()));
        ys[0].backward();
        assert_eq!(*x.get_grad().unwrap(), 1.0.into());
        assert_eq!(*y.get_grad().unwrap(), 1.0.into());
        assert_eq!(*z.get_grad().unwrap(), 1.0.into());
    }
    {
        let x = Variable::new(3.0.into());
        Sum.call(vec![x.clone(), x.clone()]);
        let ys = Sum.call(vec![x.clone(), x.clone()]);
        assert_eq!(*ys[0], 6.0.into());

        ys[0].set_grad(Variable::new(1.0.into()));
        ys[0].backward();
        assert_eq!(*x.get_grad().unwrap(), 2.0.into());
    }
}

#[test]
fn test_sub() {
    let a = Variable::new(5.0.into());
    let b = Variable::new(3.0.into());
    let ys = Sub.call(vec![a.clone(), b.clone()]);
    assert_eq!(*ys[0], 2.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward();
    assert_eq!(*a.get_grad().unwrap(), 1.0.into());
    assert_eq!(*b.get_grad().unwrap(), (-1.0).into());
}
