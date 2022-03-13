use crate::{functions::Neg, Function, Tensor, Variable};

pub struct Sub;

impl Function for Sub {
    fn forward<const ENABLE_BACKPROP: bool>(
        &self,
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 2);
        assert_eq!(xs[0].shape(), xs[1].shape());

        vec![&*xs[0] - &*xs[1]]
    }

    fn backward<const ENABLE_BACKPROP: bool>(
        &self,
        _xs: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        vec![
            gys[0].clone(),
            Neg.call(vec![gys[0].clone()]).pop().unwrap(),
        ]
    }
}

#[test]
fn test_sub() {
    let a = Variable::<true>::new(ndarray::arr0(5.0).into_dyn());
    let b = Variable::new(ndarray::arr0(3.0).into_dyn());
    let ys = Sub.call(vec![a.clone(), b.clone()]);
    assert_eq!(*ys[0], ndarray::arr0(2.0).into_dyn());

    ys[0].set_grad(Variable::<true>::new(ndarray::arr0(1.0).into_dyn()));
    ys[0].backward(false, false);
    assert_eq!(
        *a.get_grad::<false>().unwrap(),
        ndarray::arr0(1.0).into_dyn()
    );
    assert_eq!(
        *b.get_grad::<false>().unwrap(),
        (ndarray::arr0(-1.0).into_dyn())
    );
}
