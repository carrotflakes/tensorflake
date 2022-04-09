use crate::*;

pub fn t(x: &Tensor) -> Tensor {
    let y = Tensor::new((**x).t().into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "t",
        move |_xs, _ys, gys| {
            let gx = gys[0].t();
            vec![gx]
        },
    );

    y
}

pub struct T;

impl Function for T {
    fn forward(&self, xs: &[Tensor]) -> Vec<Tensor> {
        assert!(xs.len() == 1);

        vec![(&*xs[0]).t().into_ndarray().into()]
    }

    fn backward(&self, xs: &Vec<Tensor>, ys: &Vec<Tensor>, gys: &Vec<Tensor>) -> Vec<Tensor> {
        #![allow(unused_variables)]

        T.call(vec![gys[0].clone()])
    }
}

#[test]
fn test() {
    {
        let x = Tensor::new(ndarray::array![[1., 2., 3.], [4., 5., 6.]].into_ndarray());
        let ys = T.call(vec![x.clone()]);
        assert_eq!(&ys[0].shape(), &[3, 2]);
    }

    {
        let x = Tensor::new(ndarray::array![[[1., 2., 3.], [4., 5., 6.]]].into_ndarray());
        let ys = T.call(vec![x.clone()]);
        assert_eq!(&ys[0].shape(), &[3, 2, 1]);
    }
}
