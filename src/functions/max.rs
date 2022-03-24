use ndarray::{prelude::*, Zip};

use crate::*;

pub fn max(axis: usize, x: &Variable) -> Variable {
    let y: Variable = x
        .map_axis(Axis(axis), |x| {
            x.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
        })
        .into_tensor()
        .into();

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "max",
        move |xs, ys, gys| {
            let shape = max_backward_shape(&xs[0], &[axis]);
            let mut mask = (*xs[0]).to_owned();
            Zip::from(&mut mask)
                .and(
                    &ys[0]
                        .view()
                        .into_shape(shape.clone())
                        .unwrap()
                        .broadcast(xs[0].shape().to_vec())
                        .unwrap(),
                )
                .for_each(|m, y| *m = if *m == *y { 1.0 } else { 0.0 });
            vec![&Variable::new(mask.into_tensor()) * &gys[0].reshape(shape.clone())]
        },
    );
    y
}

fn max_backward_shape(x: &Tensor, axes: &[usize]) -> Vec<usize> {
    x.shape()
        .iter()
        .enumerate()
        .map(|(i, &s)| if axes.contains(&i) { 1 } else { s })
        .collect()
}

#[test]
fn test() {
    let x = backprop(array![[1., 2.], [3., 4.]].into_tensor());
    let y = max(0, &x);
    assert_eq!(&*y, &array![3., 4.].into_tensor());

    let grads = gradients(&[y], &[x.clone()], false);
    assert_eq!(&*grads[0], &array![[0., 0.], [1., 1.]].into_tensor());

    let y = max(1, &x);
    assert_eq!(&*y, &array![2., 4.].into_tensor());

    let grads = gradients(&[y], &[x.clone()], false);
    assert_eq!(&*grads[0], &array![[0., 1.], [0., 1.]].into_tensor());
}
