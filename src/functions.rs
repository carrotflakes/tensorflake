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
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = &y + &x;
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
        let mut data = xs[0].data.clone();
        for x in xs.iter().skip(1) {
            for (a, b) in data.iter_mut().zip(&x.data) {
                *a *= *b;
            }
        }
        vec![Tensor::new(data, &xs[0].shape)]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        (0..xs.len())
            .map(|i| {
                let mut data = gys[0].data.clone();
                for j in 0..xs.len() {
                    if j != i {
                        for (a, b) in data.iter_mut().zip(&xs[j].data) {
                            *a *= *b;
                        }
                    }
                }
                Variable::new(Tensor::new(data, &xs[0].shape))
            })
            .collect()
    }
}

pub struct Sub;

impl Function for Sub {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 2);
        assert_eq!(xs[0].shape, xs[1].shape);

        vec![Tensor::new(
            xs[0]
                .data
                .iter()
                .zip(&xs[1].data)
                .map(|(a, b)| a - b)
                .collect(),
            &xs[0].shape,
        )]
    }

    fn backward(&self, _xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        vec![
            gys[0].clone(),
            Variable::new(gys[0].multiply_with_scalar(-1.0)),
        ]
    }
}

pub struct Div;

impl Function for Div {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 2);
        assert_eq!(xs[0].shape, xs[1].shape);

        vec![Tensor::new(
            xs[0]
                .data
                .iter()
                .zip(&xs[1].data)
                .map(|(a, b)| a / b)
                .collect(),
            &xs[0].shape,
        )]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        let mut gx0 = gys[0].data.clone();
        let mut gx1 = gys[0].data.clone();
        let x0 = &xs[0].data;
        let x1 = &xs[1].data;
        for i in 0..gx0.len() {
            gx0[i] = gx0[i] / x1[i];
            gx1[i] = gx1[i] * (-x0[i] / x1[i].powi(2));
        }
        vec![
            Variable::new(Tensor::new(gx0, &gys[0].shape)),
            Variable::new(Tensor::new(gx1, &gys[0].shape)),
        ]
    }
}

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

impl Function for Pow {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![Tensor::new(
            xs[0].data.iter().map(|a| a.powf(self.0)).collect(),
            &xs[0].shape,
        )]
    }

    fn backward(&self, xs: &Vec<Variable>, gys: &Vec<Variable>) -> Vec<Variable> {
        let mut gx = gys[0].data.clone();
        let x0 = &xs[0].data;
        for i in 0..gx.len() {
            gx[i] = gx[i] * self.0 * x0[i].powf(self.0 - 1.0);
        }
        vec![Variable::new(Tensor::new(gx, &gys[0].shape))]
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

#[test]
fn test_pow() {
    let a = Variable::new(5.0.into());
    let ys = Pow(2.0).call(vec![a.clone()]);
    assert_eq!(*ys[0], 25.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward();
    assert_eq!(*a.get_grad().unwrap(), 10.0.into());
}
