use crate::*;

use super::{sum_to_axes_to_desire, SumTo};

pub struct Add;

impl Function for Add {
    fn forward(&self, xs: &Vec<Variable>) -> Vec<Variable> {
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
                    gy = call!(SumTo::new(sum_to_axes_to_desire(gy.shape(), x.shape())), gy);
                }

                gy
            })
            .collect()
    }
}

// #[test]
// fn test_add() {
//     use crate::scalar;

//     {
//         let x = Variable::new(scalar(1.0));
//         let y = Variable::new(scalar(2.0));
//         let z = Variable::new(scalar(3.0));
//         let xs = vec![x.clone(), y.clone(), z.clone()];
//         let ys = Add.call(xs);
//         assert_eq!(*ys[0], scalar(6.0));

//         ys[0].backward(false, false);
//         assert_eq!(*x.get_grad().unwrap(), scalar(1.0));
//         assert_eq!(*y.get_grad().unwrap(), scalar(1.0));
//         assert_eq!(*z.get_grad().unwrap(), scalar(1.0));
//     }
//     {
//         let x = Variable::new(scalar(3.0));
//         Add.call(vec![x.clone(), x.clone()]);
//         let ys = Add.call(vec![x.clone(), x.clone()]);
//         assert_eq!(*ys[0], scalar(6.0));

//         ys[0].backward(false, false);
//         assert_eq!(*x.get_grad().unwrap(), scalar(2.0));
//     }
// }
