use ndarray::{ArrayBase, CowRepr, Dimension, OwnedRepr, ViewRepr};

pub type Tensor = ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>;

pub fn scalar(x: f32) -> Tensor {
    ndarray::arr0(x).into_tensor()
}

pub trait IntoTensor {
    fn into_tensor(self) -> Tensor;
}

impl<D: Dimension> IntoTensor for ArrayBase<OwnedRepr<f32>, D> {
    fn into_tensor(self) -> Tensor {
        self.into_dyn().into_owned()
    }
}

impl<D: Dimension> IntoTensor for ArrayBase<ViewRepr<&f32>, D> {
    fn into_tensor(self) -> Tensor {
        self.into_dyn().to_owned()
    }
}

impl<'a, D: Dimension> IntoTensor for ArrayBase<CowRepr<'a, f32>, D> {
    fn into_tensor(self) -> Tensor {
        self.into_dyn().to_owned()
    }
}
