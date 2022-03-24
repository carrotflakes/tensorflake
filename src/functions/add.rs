use crate::*;

use super::{sum_axes_to_desire, Sum};

pub struct Add;

impl Function for Add {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() >= 1);
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = y + &**x;
        }
        vec![Variable::new(y)]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        xs.iter()
            .map(|x| {
                let mut gy = gys[0].clone();

                // fit shape
                if x.shape() != gy.shape() {
                    gy = call!(
                        Sum::new(sum_axes_to_desire(gy.shape(), x.shape()), false),
                        gy
                    );
                }

                gy
            })
            .collect()
    }
}

#[test]
fn test_add() {
    use crate::scalar;

    {
        let x = backprop(scalar(1.0));
        let y = backprop(scalar(2.0));
        let z = backprop(scalar(3.0));
        let xs = vec![x.clone(), y.clone(), z.clone()];
        let ys = Add.call(xs);
        assert_eq!(*ys[0], scalar(6.0));

        let grads = gradients(&ys, &vec![x.clone(), y.clone(), z.clone()], false);
        assert_eq!(grads[0][[]], 1.0);
        assert_eq!(grads[1][[]], 1.0);
        assert_eq!(grads[2][[]], 1.0);
    }
    {
        let x = backprop(scalar(3.0));
        Add.call(vec![x.clone(), x.clone()]);
        let ys = Add.call(vec![x.clone(), x.clone()]);
        assert_eq!(*ys[0], scalar(6.0));

        let grads = gradients(&ys, &vec![x.clone()], false);
        assert_eq!(grads[0][[]], 2.0);
    }
}
