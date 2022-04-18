use crate::{
    functions::{sum_axes_to_desire, Neg, Sum},
    *,
};

pub fn sub(lhs: &Computed, rhs: &Computed) -> Computed {
    let y = Computed::new((&**lhs - &**rhs).into_ndarray());

    chain(
        &[lhs.clone(), rhs.clone()],
        &[y.clone()],
        false,
        "sub",
        |xs, _ys, gys| {
            let mut gx0 = gys[0].clone();
            let mut gx1 = -&gys[0];

            // fit shape
            if xs[0].shape() != gx0.shape() {
                gx0 = gx0.sum(sum_axes_to_desire(gx0.shape(), xs[0].shape()), false);
            }

            if xs[1].shape() != gx1.shape() {
                gx1 = gx1.sum(sum_axes_to_desire(gx1.shape(), xs[0].shape()), false);
            }

            vec![gx0, gx1]
        },
    );

    y
}

pub struct Sub;

impl Function for Sub {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 2);

        vec![(&*xs[0] - &*xs[1]).into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
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
