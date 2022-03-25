use ndarray::{ArrayBase, CowRepr, Dim, Dimension, OwnedRepr, ViewRepr};

pub type NDArray = ArrayBase<ndarray::OwnedArcRepr<f32>, ndarray::IxDyn>;

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

pub fn as_2d(tensor: &NDArray) -> ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>> {
    let shape = tensor.shape();
    tensor
        .view()
        .into_shape([
            shape.iter().take(shape.len() - 1).product(),
            *shape.last().unwrap(),
        ])
        .unwrap()
}
