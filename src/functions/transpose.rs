use crate::*;

pub struct Transpose {
    axes: Vec<usize>,
}

impl Transpose {
    pub fn new(axes: Vec<usize>) -> Self {
        assert!((0..axes.len()).all(|i| axes.contains(&i)));

        Self { axes }
    }
}

impl Function for Transpose {
    fn forward(&self, xs: &[Variable]) -> Vec<Variable> {
        assert!(xs.len() == 1);

        vec![xs[0].view().permuted_axes(&*self.axes).into_tensor().into()]
    }

    fn backward(
        &self,
        xs: &Vec<Variable>,
        ys: &Vec<Variable>,
        gys: &Vec<Variable>,
    ) -> Vec<Variable> {
        #![allow(unused_variables)]

        Transpose::new(
            (0..self.axes.len())
                .map(|i| self.axes.iter().position(|j| *j == i).unwrap())
                .collect::<Vec<_>>(),
        )
        .call(vec![gys[0].clone()])
    }
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::Array::zeros([1, 2, 3]).into_tensor());
        let y = call!(Transpose::new(vec![1, 2, 0]), x);
        assert_eq!(y.shape(), &[2, 3, 1]);

        let grads = gradients(&[y], &[x], false);
        assert_eq!(grads[0].shape(), &[1, 2, 3]);
    }
}
