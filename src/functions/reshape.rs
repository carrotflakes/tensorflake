use crate::*;

pub struct Reshape {
    pub shape: Vec<usize>,
    original_shape: Vec<usize>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            original_shape: Vec::new(),
        }
    }
}

impl Function for Reshape {
    fn forward(&self, xs: &Vec<crate::Variable>) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0]
            .to_shape(self.shape.as_slice())
            .unwrap()
            .into_tensor()
            .into()]
    }

    fn backward(
        &self,
        xs: &Vec<crate::Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<crate::Variable>,
    ) -> Vec<crate::Variable> {
        #![allow(unused_variables)]

        vec![Variable::new(
            gys[0]
                .broadcast(self.shape.as_slice())
                .unwrap()
                .to_shape(self.original_shape.as_slice())
                .unwrap()
                .into_tensor(),
        )]
    }

    fn into_backward(mut self, xs: &Vec<crate::Variable>) -> Box<dyn crate::Backward>
    where
        Self: Sized + 'static,
    {
        self.original_shape = xs[0].shape().to_vec();
        Box::new(self)
    }
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_tensor());
        let ys = Reshape::new(vec![3, 2]).call(vec![x.clone()]);
        dbg!(&*ys[0]);
        assert_eq!(ys[0].shape(), &[3, 2]);

        let grads = gradients(&ys, &[x.clone()], false);
        assert_eq!(grads[0].shape(), &[2, 3]);
    }
}
