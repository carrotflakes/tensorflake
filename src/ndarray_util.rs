use ndarray::{
    Array, ArrayBase, Axis, CowRepr, Dim, Dimension, OwnedArcRepr, OwnedRepr, RemoveAxis, ViewRepr,
};

pub type NDArray = ArrayBase<OwnedArcRepr<f32>, ndarray::IxDyn>;

pub fn scalar(x: f32) -> NDArray {
    ndarray::arr0(x).into_ndarray()
}

pub trait IntoNDArray {
    fn into_ndarray(self) -> NDArray;
}

impl<D: Dimension> IntoNDArray for ArrayBase<OwnedRepr<f32>, D> {
    fn into_ndarray(self) -> NDArray {
        self.into_dyn().into_shared()
    }
}

impl<D: Dimension> IntoNDArray for ArrayBase<ViewRepr<&f32>, D> {
    fn into_ndarray(self) -> NDArray {
        self.into_dyn().to_shared()
    }
}

impl<'a, D: Dimension> IntoNDArray for ArrayBase<CowRepr<'a, f32>, D> {
    fn into_ndarray(self) -> NDArray {
        self.into_dyn().to_shared()
    }
}

pub fn as_2d(tensor: &NDArray) -> ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>> {
    let shape = tensor.shape();
    tensor
        .view()
        .into_shape([
            shape.iter().take(shape.len() - 1).product(),
            *shape.last().unwrap(),
        ])
        .unwrap()
}

pub fn onehot<D: Dimension>(t: &Array<usize, D>, size: usize) -> NDArray {
    let mut v = vec![0.0; t.shape().iter().product::<usize>() * size];
    for (i, n) in t.iter().copied().enumerate() {
        v[i * size + n] = 1.0;
    }
    ndarray::Array::from_shape_vec(
        t.shape().iter().cloned().chain([size]).collect::<Vec<_>>(),
        v,
    )
    .unwrap()
    .into_ndarray()
}

pub fn argmax<D: Dimension + RemoveAxis>(
    t: &ArrayBase<OwnedArcRepr<f32>, D>,
) -> Array<usize, D::Smaller> {
    t.map_axis(Axis(t.ndim() - 1), |x| {
        x.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    })
}

#[test]
fn test_argmax() {
    let t = ndarray::arr2(&[[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]).into_ndarray();
    assert_eq!(argmax(&t).into_raw_vec(), [2, 0]);
}
