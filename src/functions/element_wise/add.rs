use crate::*;

use super::super::sum_axes_to_desire;

pub fn add(a: &ComputedNDA, b: &ComputedNDA) -> ComputedNDA {
    let y = ComputedNDA::new((&**a + &**b).into_ndarray());

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

pub fn multi_add(xs: &[ComputedNDA]) -> ComputedNDA {
    let mut y = (*xs[0]).clone();
    for x in xs.iter().skip(1) {
        y = y + &**x;
    }
    let y = ComputedNDA::new(y);

    chain(xs, &[y.clone()], false, "multi_add", |xs, _ys, gys| {
        xs.iter()
            .map(|x| {
                let mut gx = gys[0].clone();

                // fit shape
                if x.shape() != gx.shape() {
                    gx = gx.sum(sum_axes_to_desire(gx.shape(), x.shape()), false);
                }

                gx
            })
            .collect()
    });

    y
}

#[test]
fn test() {
    use crate::scalar;

    {
        let a = backprop(scalar(1.0));
        let b = backprop(scalar(2.0));
        let c = backprop(scalar(3.0));
        let y = multi_add(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(*y, scalar(6.0));

        let grads = gradients(&[y], &[a.clone(), b.clone(), c.clone()], false);
        assert_eq!(grads[0][[]], 1.0);
        assert_eq!(grads[1][[]], 1.0);
        assert_eq!(grads[2][[]], 1.0);
    }
    {
        let x = backprop(scalar(3.0));
        let y = multi_add(&[x.clone(), x.clone()]);
        assert_eq!(*y, scalar(6.0));

        let grads = gradients(&[y], &[x.clone()], false);
        assert_eq!(grads[0][[]], 2.0);
    }
}
