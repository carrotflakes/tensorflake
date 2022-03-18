use super::Mul;
use crate::*;

pub struct Pow(f32);

impl Pow {
    pub fn new(x: f32) -> Pow {
        Pow(x)
    }
}

impl Function for Pow {
    fn forward(
        &self,
        xs: &Vec<Variable>,
    ) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![xs[0].map(|x| x.powf(self.0)).into_tensor()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        Mul.call(vec![
            Pow::new(self.0 - 1.0)
                .call(vec![xs[0].clone()])
                .pop()
                .unwrap(),
            gys[0].clone(),
            Variable::new(scalar(self.0)),
        ])
    }
}

// #[test]
// fn test_pow() {
//     let a = Variable::new(scalar(5.0));
//     let ys = Pow(2.0).call(vec![a.clone()]);
//     assert_eq!(*ys[0], scalar(25.0));

//     ys[0].backward(false, false);
//     assert_eq!(*a.get_grad().unwrap(), scalar(10.0));
// }
