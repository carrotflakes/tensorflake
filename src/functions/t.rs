use crate::*;

pub struct T;

impl Function for T {
    fn forward(
        &self,
        xs: &Vec<Variable>,
    ) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].t().into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        T.call(vec![gys[0].clone()])
    }
}

// #[test]
// fn test() {
//     use crate::{Variable, ENABLE_BACKPROP};

//     {
//         let x = Variable::new(
//             ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_dyn(),
//         );
//         let ys = T.call(vec![x.clone()]);
//         assert_eq!(&ys[0].shape(), &[3, 2]);
//     }

//     {
//         let x = Variable::new(
//             ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_dyn(),
//         );
//         let ys = T.call(vec![x.clone()]);
//         assert_eq!(&ys[0].shape(), &[3, 2, 1]);
//     }
// }
