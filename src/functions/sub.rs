use crate::{
    functions::{sum_axes_to_desire, Neg, Sum},
    *,
};

pub struct Sub;

impl Function for Sub {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 2);

        vec![(&*xs[0] - &*xs[1]).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        let mut gx0 = gys[0].clone();
        let mut gx1 = Neg.call(vec![gys[0].clone()]).pop().unwrap();

        // fit shape
        if xs[0].shape() != gx0.shape() {
            gx0 = call!(
                Sum::new(sum_axes_to_desire(gx0.shape(), xs[0].shape()), false),
                gx0
            );
        }

        if xs[1].shape() != gx1.shape() {
            gx1 = call!(
                Sum::new(sum_axes_to_desire(gx1.shape(), xs[0].shape()), false),
                gx1
            );
        }

        vec![gx0, gx1]
    }
}

#[test]
fn test_sub() {
    use crate::scalar;

    let a = backprop(scalar(5.0));
    let b = backprop(scalar(3.0));
    let ys = Sub.call(vec![a.clone(), b.clone()]);
    assert_eq!(*ys[0], scalar(2.0));

    let grads = gradients(&ys, &[a.clone(), b.clone()], false);
    assert_eq!(&*grads[0], scalar(1.0));
    assert_eq!(&*grads[1], scalar(-1.0));
}
