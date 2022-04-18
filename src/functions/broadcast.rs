use crate::*;

use super::Sum;

pub fn broadcast(x: &Computed, shape: impl Into<Vec<usize>>) -> Computed {
    let shape = shape.into();
    let y = Computed::new(
        (**x)
            .broadcast(shape.as_slice())
            .unwrap_or_else(|| panic!("illegal broadcast: {:?} to {:?}", x.shape(), shape))
            .into_ndarray(),
    );

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "broadcast",
        move |xs, _ys, gys| {
            let mut axes = Vec::new();
            let mut target = xs[0].shape().to_vec();
            for (axis, size) in shape.iter().enumerate() {
                if let Some(s) = target.first() {
                    if s == size {
                        target.remove(0);
                        continue;
                    }
                }
                axes.push(axis);
            }

            let gx = gys[0].sum(axes, false);

            vec![gx]
        },
    );

    y
}

pub struct Broadcast {
    pub shape: Vec<usize>,
    axes: Vec<usize>,
}

impl Broadcast {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            axes: Vec::new(),
        }
    }
}

impl Function for Broadcast {
    fn forward(&self, xs: &[Computed]) -> Vec<Computed> {
        assert!(xs.len() == 1);

        vec![Computed::new(
            (*xs[0])
                .broadcast(self.shape.as_slice())
                .unwrap_or_else(|| {
                    panic!("illegal broadcast: {:?} to {:?}", xs[0].shape(), self.shape)
                })
                .into_ndarray(),
        )]
    }

    fn backward(&self, xs: &Vec<Computed>, ys: &Vec<Computed>, gys: &Vec<Computed>) -> Vec<Computed> {
        #![allow(unused_variables)]

        vec![call!(Sum::new(self.axes.clone(), false), &gys[0])]
    }

    fn into_backward(mut self, xs: &Vec<Computed>) -> Box<dyn Backward>
    where
        Self: Sized + 'static,
    {
        // TODO: test
        let axes = &mut self.axes;
        let mut target = xs[0].shape().to_vec();
        for (axis, size) in self.shape.iter().enumerate() {
            if let Some(s) = target.first() {
                if s == size {
                    target.remove(0);
                    continue;
                }
            }
            axes.push(axis);
        }
        Box::new(self)
    }
}

#[test]
fn test() {
    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let ys = Broadcast::new(vec![2, 3]).call(vec![x.clone()]);
        assert_eq!(ys[0].shape(), &[2, 3]);
        assert_eq!(&*ys[0], &*x);
    }

    {
        let x = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let ys = Broadcast::new(vec![4, 2, 3]).call(vec![x.clone()]);
        // dbg!(&*ys[0]);
        assert_eq!(ys[0].shape(), &[4, 2, 3]);
    }
}
