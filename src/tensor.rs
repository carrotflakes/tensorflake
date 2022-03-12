use std::ops::Add;

use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: SmallVec<[usize; 4]>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());

        Self {
            data,
            shape: shape.into(),
        }
    }

    pub fn get_data(&self) -> &[f32] {
        &self.data
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let mut data = self.data.clone();
        for x in &mut data {
            *x = f(*x);
        }
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.len() == 0
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape);

        let len = self.data.len();
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.data[i] * other.data[i]);
        }
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn multiply_with_scalar(&self, rhs: f32) -> Self {
        let len = self.data.len();
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.data[i] * rhs);
        }
        Self {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let mut data = Vec::with_capacity(self.data.len());
        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            data.push(*l + *r);
        }

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl From<f32> for Tensor {
    fn from(x: f32) -> Self {
        Tensor::new(vec![x], &[])
    }
}
