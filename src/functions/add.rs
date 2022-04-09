use crate::*;

use super::{sum_axes_to_desire, Sum};

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let y = Tensor::new((&**a + &**b).into_ndarray());

    chain(
        &[a.clone(), b.clone()],
        &[y.clone()],
        false,
        "add",
        |xs, _ys, gys| {
            let mut gx1 = gys[0].clone();
            let mut gx2 = gys[0].clone();

            // fit shape
            if xs[0].shape() != gx1.shape() {
                gx1 = gx1.sum(sum_axes_to_desire(gx1.shape(), xs[0].shape()), false);
            }
            if xs[1].shape() != gx2.shape() {
                gx2 = gx2.sum(sum_axes_to_desire(gx2.shape(), xs[1].shape()), false);
            }

            vec![gx1, gx2]
        },
    );

    y
}

pub struct Add;

impl Function for Add {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() >= 1);
        let mut y = (*xs[0]).clone();
        for x in xs.iter().skip(1) {
            y = y + &**x;
        }
        vec![Tensor::new(y)]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        xs.iter()
            .map(|x| {
                let mut gx = gys[0].clone();

                // fit shape
                if x.shape() != gx.shape() {
                    gx = call!(
                        Sum::new(sum_axes_to_desire(gx.shape(), x.shape()), false),
                        gx
                    );
                }

                gx
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
