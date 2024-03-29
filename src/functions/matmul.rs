use ndarray::{Axis, Ix2};

use crate::functions::*;
use crate::*;

pub fn matmul(lhs: &ComputedNDA, rhs: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new(forward(&lhs, &rhs));

    chain(
        &[lhs.clone(), rhs.clone()],
        &[y.clone()],
        false,
        "matmul",
        |xs, _, gys| {
            let x = xs[0].clone();
            let w = xs[1].clone();
            let gx = gys[0].matmul(&w.mat_t());
            let gw = x.mat_t().matmul(&gys[0]);
            vec![gx, gw]
        },
    );

    y
}

pub fn forward(x0: &NDArray, x1: &NDArray) -> NDArray {
    // let x0 = if x0.ndim() == 1 {
    //     dbg!("a", x1.shape());
    //     // extend axis if x0 is 1d
    //     x0.clone().insert_axis(Axis(0))
    // } else {
    //     x0.clone()
    // };

    // 行列同士の積に限定する
    let x0s = x0.shape();
    let x1s = x1.shape();
    assert!(2 <= x0s.len());
    assert!(2 <= x1s.len());
    assert_eq!(
        x0s[x0s.len() - 1],
        x1s[x1s.len() - 2],
        "lhs's width must be equal to rhs's height, lhs: {:?}, rhs: {:?}",
        x0s,
        x1s
    );

    let outer_shape =
        broadcast_shape(&x0s[..x0s.len() - 2], &x1s[..x1s.len() - 2]).expect(&format!(
            "Matmul: shape mismatch: {:?} and {:?}",
            x0.shape(),
            x1.shape()
        ));
    let mat_shape = [x0s[x0s.len() - 2], x1s[x1s.len() - 1]];

    if outer_shape.is_empty() {
        let x0 = x0.to_owned().into_dimensionality::<Ix2>().unwrap();
        let x1 = x1.to_owned().into_dimensionality::<Ix2>().unwrap();

        x0.dot(&x1).into_ndarray()
    } else {
        let x0 = x0
            .broadcast(
                outer_shape
                    .iter()
                    .chain(&x0s[x0s.len() - 2..])
                    .cloned()
                    .collect::<Vec<usize>>(),
            )
            .unwrap();
        let x1 = x1
            .broadcast(
                outer_shape
                    .iter()
                    .chain(&x1s[x1s.len() - 2..])
                    .cloned()
                    .collect::<Vec<usize>>(),
            )
            .unwrap();
        let outer_size = outer_shape.iter().product();
        let mut es = Vec::with_capacity(outer_size * mat_shape.iter().product::<usize>());
        let x0 = x0
            .to_shape(
                [outer_size]
                    .iter()
                    .chain(&x0s[x0s.len() - 2..])
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x1 = x1
            .to_shape(
                [outer_size]
                    .iter()
                    .chain(&x1s[x1s.len() - 2..])
                    .cloned()
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for i in 0..outer_size {
            let x0 = x0
                .index_axis(Axis(0), i)
                .into_dimensionality::<Ix2>()
                .unwrap();
            let x1 = x1
                .index_axis(Axis(0), i)
                .into_dimensionality::<Ix2>()
                .unwrap();
            es.extend(x0.dot(&x1).into_raw_vec());
        }

        ndarray::Array::from_shape_vec(
            outer_shape
                .iter()
                .chain(mat_shape.iter())
                .cloned()
                .collect::<Vec<_>>(),
            es,
        )
        .unwrap()
        .into_ndarray()
    }
}

pub fn backward(x: &NDArray, w: &NDArray, gy: &NDArray) -> (NDArray, NDArray) {
    let gx = forward(gy, &mat_transpose::forward(w));
    let gw = forward(&mat_transpose::forward(x), gy);
    (gx, gw)
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    (0..a.len().max(b.len()))
        .rev()
        .map(|i| {
            let a = a.get(i).copied().unwrap_or(1);
            let b = b.get(i).copied().unwrap_or(1);
            if a == b || b == 1 {
                Some(a)
            } else if a == 1 {
                Some(b)
            } else {
                None
            }
        })
        .rev()
        .collect()
}

#[test]
fn test() {
    {
        let a = backprop(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let b = backprop(ndarray::array![[1., 2.], [3., 4.], [5., 6.]].into_ndarray());
        let y = matmul(&a, &b);
        assert_eq!(&y.shape(), &[2, 2]);

        let _grads = gradients(&[y], &[a.clone(), b.clone()], false);

        let y = matmul(&b, &a);
        assert_eq!(&y.shape(), &[3, 3]);
    }

    {
        let a = backprop(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
        let b = backprop(ndarray::array![[[1., 2.], [3., 4.], [5., 6.]]].into_ndarray());
        let y = matmul(&a, &b);
        assert_eq!(&y.shape(), &[1, 2, 2]);

        let _grads = gradients(&[y], &[a.clone(), b.clone()], false);

        let y = matmul(&b, &a);
        assert_eq!(&y.shape(), &[1, 3, 3]);
    }

    {
        let a = backprop(
            NDArray::from_shape_vec(&[2, 3, 4, 5][..], (0..120).map(|x| x as f32).collect())
                .unwrap(),
        );
        let b = backprop(
            NDArray::from_shape_vec(&[2, 3, 5, 4][..], (0..120).map(|x| x as f32).collect())
                .unwrap(),
        );
        let y = matmul(&a, &b);
        assert_eq!(&y.shape(), &[2, 3, 4, 4]);

        let _grads = gradients(&[y], &[a.clone(), b.clone()], false);

        let y = matmul(&b, &a);
        assert_eq!(&y.shape(), &[2, 3, 5, 5]);
    }
}
