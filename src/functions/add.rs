use crate::{Function, Tensor, Variable};

pub struct Add;

impl Function for Add {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = y + &**x;
        }
        vec![y]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        (0..xs.len()).map(|_| gys[0].clone()).collect()
    }
}

#[test]
fn test_add() {
    {
        let x = Variable::<true>::new(ndarray::arr0(1.0).into_dyn());
        let y = Variable::new(ndarray::arr0(2.0).into_dyn());
        let z = Variable::new(ndarray::arr0(3.0).into_dyn());
        let xs = vec![x.clone(), y.clone(), z.clone()];
        let ys = Add.call(xs);
        assert_eq!(*ys[0], ndarray::arr0(6.0).into_dyn());

        ys[0].set_grad(Variable::<true>::new(ndarray::arr0(1.0).into_dyn()));
        ys[0].backward(false, false);
        assert_eq!(
            *x.get_grad::<false>().unwrap(),
            ndarray::arr0(1.0).into_dyn()
        );
        assert_eq!(
            *y.get_grad::<false>().unwrap(),
            ndarray::arr0(1.0).into_dyn()
        );
        assert_eq!(
            *z.get_grad::<false>().unwrap(),
            ndarray::arr0(1.0).into_dyn()
        );
    }
    {
        let x = Variable::<true>::new(ndarray::arr0(3.0).into_dyn());
        Add.call(vec![x.clone(), x.clone()]);
        let ys = Add.call(vec![x.clone(), x.clone()]);
        assert_eq!(*ys[0], ndarray::arr0(6.0).into_dyn());

        ys[0].set_grad(Variable::<true>::new(ndarray::arr0(1.0).into_dyn()));
        ys[0].backward(false, false);
        assert_eq!(
            *x.get_grad::<false>().unwrap(),
            ndarray::arr0(2.0).into_dyn()
        );
    }
}
