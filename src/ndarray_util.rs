use ndarray::{
    Array, ArrayBase, ArrayView1, ArrayView2, Axis, CowArray, CowRepr, Data, Dim, Dimension, Ix1,
    Ix2, IxDyn, OwnedArcRepr, OwnedRepr, RemoveAxis, ViewRepr,
};

pub use ndarray_einsum_beta::tensordot;

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

pub fn as_2d(array: &NDArray) -> ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>> {
    let shape = array.shape();
    array
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN"))
            .unwrap()
            .0
    })
}

#[test]
fn test_argmax() {
    let t = ndarray::arr2(&[[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]]).into_ndarray();
    assert_eq!(argmax(&t).into_raw_vec(), [2, 0]);
}

pub fn map_ex_axis<'a, A, S, D, B, F>(
    array: &'a ArrayBase<S, D>,
    axis: Axis,
    mapping: F,
) -> Array<B, Ix1>
where
    A: 'a,
    S: Data<Elem = A>,
    D: RemoveAxis,
    F: FnMut(ArrayView1<'_, A>) -> B,
{
    let axis_size = array.shape()[axis.0];
    let mut a = array.view();
    a.swap_axes(axis.0, a.ndim() - 1);
    let b = a.into_shape([array.len() / axis_size, axis_size]).unwrap();
    b.map_axis(Axis(1), mapping)
}

#[test]
fn test_map_ex_axis() {
    let x = ndarray::array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]].into_ndarray();
    let y = map_ex_axis(&x, Axis(1), |x| x.mean().unwrap());
    assert_eq!(y, ndarray::array![2.5, 3.5, 4.5]);
}

pub fn map_ex_axis_keep_dim<'a, A, S, D, B, F>(
    array: &'a ArrayBase<S, D>,
    axis: Axis,
    mut mapping: F,
) -> Array<B, IxDyn>
where
    A: 'a,
    B: Clone,
    S: Data<Elem = A>,
    D: RemoveAxis,
    F: FnMut(ArrayView2<'_, A>) -> Array<B, Ix1>,
{
    let axis_size = array.shape()[axis.0];
    let mut a = array.view();
    a.swap_axes(axis.0, a.ndim() - 1);
    let b = a.into_shape([array.len() / axis_size, axis_size]).unwrap();
    let c = mapping(b);
    let mut s = array.shape().to_vec();
    for (i, a) in s.iter_mut().enumerate() {
        if i == axis.0 {
            *a = 1;
        }
    }
    c.into_shape(&s[..]).unwrap()
}

#[test]
fn test_map_ex_axis_keep_dim() {
    let x = ndarray::array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]].into_ndarray();
    let y = map_ex_axis_keep_dim(&x, Axis(1), |x| x.mean_axis(Axis(1)).unwrap());
    assert_eq!(y, ndarray::array![[[2.5, 3.5, 4.5]]].into_dyn());
}

pub fn map_axes_keep_dim<'a, A, S, D, B, F>(
    array: &'a ArrayBase<S, D>,
    axes: &[usize],
    mut mapping: F,
) -> Array<B, IxDyn>
where
    A: 'a + Clone,
    B: Clone,
    S: Data<Elem = A>,
    D: RemoveAxis,
    F: FnMut(CowArray<A, Ix2>) -> Array<B, Ix1>,
{
    let axis_size = axes.iter().map(|a| array.shape()[*a]).product::<usize>();
    let outer_shape: Vec<_> = array
        .shape()
        .iter()
        .enumerate()
        .filter(|&(i, _)| !axes.contains(&i))
        .map(|(_, a)| *a)
        .collect();
    let a = array.view().into_dyn();

    // permute axes to [outer.., inner..]
    let permutation = (0..array.ndim())
        .filter(|a| !axes.contains(a))
        .chain(axes.iter().copied())
        .collect::<Vec<_>>();
    let a = a.permuted_axes(&permutation[..]);

    // reshape to [outer, inner]
    let b = a.to_shape([array.len() / axis_size, axis_size]).unwrap();

    // mapping to [outer]
    let c = mapping(b);

    // reshape to [outer..]
    let c = c.into_shape(outer_shape).unwrap();

    // permute axes back to original order
    let mut s = array.shape().to_vec();
    for (i, a) in s.iter_mut().enumerate() {
        if axes.contains(&i) {
            *a = 1;
        }
    }
    c.into_shape(&s[..]).unwrap()
}

#[test]
fn test_map_axes_keep_dim() {
    let x =
        ndarray::Array::from_shape_vec((2, 3, 4, 5, 6), (0..2 * 3 * 4 * 5 * 6).collect()).unwrap();
    let y = map_axes_keep_dim(&x, &[1, 3], |x| x.mean_axis(Axis(1)).unwrap());
    assert_eq!(y.shape(), &[2, 1, 4, 1, 6]);
}

pub struct NDArraySummary {
    pub shape: Vec<usize>,
    pub mean: f32,
    pub var: f32,
}

impl NDArraySummary {
    pub fn from(a: &NDArray) -> Self {
        Self {
            shape: a.shape().to_vec(),
            mean: a.mean().unwrap(),
            var: a.var(1.0),
        }
    }
}

impl std::fmt::Debug for NDArraySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shape: {:?}\nmean: {:.4}\nvar: {:.4}",
            self.shape, self.mean, self.var,
        )
    }
}
