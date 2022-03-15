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
        xs: &Vec<Variable<ENABLE_BACKPROP>>,
        ys: &Vec<Variable<ENABLE_BACKPROP>>,
        gys: &Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>> {
        #![allow(unused_variables)]

        vec![
            gys[0].clone(),
            Neg.call(vec![gys[0].clone()]).pop().unwrap(),
        ]
    }
}

#[test]
fn test_sub() {
    use crate::scalar;

    let a = Variable::<true>::new(scalar(5.0));
    let b = Variable::new(scalar(3.0));
    let ys = Sub.call(vec![a.clone(), b.clone()]);
    assert_eq!(*ys[0], scalar(2.0));

    ys[0].set_grad(Variable::<true>::new(scalar(1.0)));
    ys[0].backward(false, false);
    assert_eq!(*a.get_grad::<false>().unwrap(), scalar(1.0));
    assert_eq!(*b.get_grad::<false>().unwrap(), scalar(-1.0));
}
