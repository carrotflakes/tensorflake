use crate::{Function, Tensor, Variable};

pub struct Exp;

impl Function for Exp {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| x.exp())]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        Mul.call(vec![gys[0].clone(), Exp.call(xs.clone()).pop().unwrap()])
        // vec![Variable::new(gys[0].multiply(&xs[0].map(|x| x.exp())))]
    }
}

pub struct Add;

impl Function for Add {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = &y + &x;
        }
        vec![y]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        (0..xs.len()).map(|_| gys[0].clone()).collect()
    }
}

pub struct Mul;

impl Function for Mul {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut data = xs[0].data.clone();
        for x in xs.iter().skip(1) {
            for (a, b) in data.iter_mut().zip(&x.data) {
                *a *= *b;
            }
        }
        vec![Tensor::new(data, &xs[0].shape)]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        (0..xs.len())
            .map(|i| {
                Mul.call(
                    (0..xs.len())
                        .filter(|j| *j != i)
                        .map(|j| xs[j].clone())
                        .chain(gys.iter().cloned())
                        .collect(),
                )
                .pop()
                .unwrap()
            })
            .collect()
    }
}

pub struct Neg;

impl Function for Neg {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);
        vec![xs[0].map(|x| -x)]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        _xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        Neg.call(gys.clone())
    }
}

pub struct Sub;

impl Function for Sub {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
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

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        _xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        vec![gys[0].clone(), Neg.call(vec![gys[0].clone()]).pop().unwrap()]
    }
}

pub struct Div;

impl Function for Div {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
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

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        vec![
            Div.call(vec![gys[0].clone(), gys[1].clone()])
                .pop()
                .unwrap(),
            Mul.call(vec![
                gys[0].clone(),
                Div.call(vec![
                    Neg.call(vec![xs[0].clone()]).pop().unwrap(),
                    Pow::new(2.0).call(vec![xs[1].clone()]).pop().unwrap(),
                ])
                .pop()
                .unwrap(),
            ])
            .pop()
            .unwrap(),
        ]
        // let mut gx0 = gys[0].data.clone();
        // let mut gx1 = gys[1].data.clone();
        // let x0 = &xs[0].data;
        // let x1 = &xs[1].data;
        // for i in 0..gx0.len() {
        //     gx0[i] = gx0[i] / x1[i];
        //     gx1[i] = gx1[i] * (-x0[i] / x1[i].powi(2));
        // }
        // vec![
        //     Variable::new(Tensor::new(gx0, &gys[0].shape)),
        //     Variable::new(Tensor::new(gx1, &gys[0].shape)),
        // ]
    }
}

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

impl Function for Pow {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![Tensor::new(
            xs[0].data.iter().map(|a| a.powf(self.0)).collect(),
            &xs[0].shape,
        )]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        // TODO
        let mut gx = gys[0].data.clone();
        let x0 = &xs[0].data;
        for i in 0..gx.len() {
            gx[i] = gx[i] * self.0 * x0[i].powf(self.0 - 1.0);
        }
        vec![Variable::new(Tensor::new(gx, &gys[0].shape))]
    }
}

pub struct Sin;

impl Function for Sin {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![Tensor::new(
            xs[0].data.iter().map(|a| a.sin()).collect(),
            &xs[0].shape,
        )]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        Mul.call(vec![gys[0].clone(), Cos.call(xs.clone()).pop().unwrap()])
        // vec![Variable::new(gys[0].multiply(&xs[0].map(|x| x.cos())))]
    }
}
pub struct Cos;

impl Function for Cos {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![Tensor::new(
            xs[0].data.iter().map(|a| a.cos()).collect(),
            &xs[0].shape,
        )]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        Mul.call(vec![
            gys[0].clone(),
            Neg.call(Sin.call(xs.clone())).pop().unwrap(),
        ])
    }
}

#[test]
fn test_sum() {
    {
        let x = Variable::<true>::new(1.0.into());
        let y = Variable::new(2.0.into());
        let z = Variable::new(3.0.into());
        let xs = vec![x.clone(), y.clone(), z.clone()];
        let ys = Add.call(xs);
        assert_eq!(*ys[0], 6.0.into());

        ys[0].set_grad(Variable::<true>::new(1.0.into()));
        ys[0].backward(false, false);
        assert_eq!(*x.get_grad::<false>().unwrap(), 1.0.into());
        assert_eq!(*y.get_grad::<false>().unwrap(), 1.0.into());
        assert_eq!(*z.get_grad::<false>().unwrap(), 1.0.into());
    }
    {
        let x = Variable::<true>::new(3.0.into());
        Add.call(vec![x.clone(), x.clone()]);
        let ys = Add.call(vec![x.clone(), x.clone()]);
        assert_eq!(*ys[0], 6.0.into());

        ys[0].set_grad(Variable::<true>::new(1.0.into()));
        ys[0].backward(false, false);
        assert_eq!(*x.get_grad::<false>().unwrap(), 2.0.into());
    }
}

#[test]
fn test_sub() {
    let a = Variable::<true>::new(5.0.into());
    let b = Variable::new(3.0.into());
    let ys = Sub.call(vec![a.clone(), b.clone()]);
    assert_eq!(*ys[0], 2.0.into());

    ys[0].set_grad(Variable::<true>::new(1.0.into()));
    ys[0].backward(false, false);
    assert_eq!(*a.get_grad::<false>().unwrap(), 1.0.into());
    assert_eq!(*b.get_grad::<false>().unwrap(), (-1.0).into());
}

#[test]
fn test_pow() {
    let a = Variable::<true>::new(5.0.into());
    let ys = Pow(2.0).call(vec![a.clone()]);
    assert_eq!(*ys[0], 25.0.into());

    ys[0].set_grad(Variable::<true>::new(1.0.into()));
    ys[0].backward(false, false);
    assert_eq!(*a.get_grad::<false>().unwrap(), 10.0.into());
}
