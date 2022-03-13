use std::{ops::Add, rc::Rc};

use smallvec::SmallVec;

#[derive(Clone, PartialEq)]
pub struct Tensor {
    pub(crate) data: Rc<Vec<f32>>,
    pub(crate) shape: SmallVec<[usize; 4]>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());

        Self {
            data: Rc::new(data),
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
        let mut data = self.data.as_ref().clone();
        for x in &mut data {
            *x = f(*x);
        }
        Self {
            data: Rc::new(data),
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
            data: Rc::new(data),
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
            data: Rc::new(data),
            shape: self.shape.clone(),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), self.data.len());

        Self {
            data: self.data.clone(),
            shape: shape.into(),
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
            data: Rc::new(data),
            shape: self.shape.clone(),
        }
    }
}

impl From<f32> for Tensor {
    fn from(x: f32) -> Self {
        Tensor::new(vec![x], &[])
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t: Vec<usize> = self
            .shape
            .iter()
            .rev()
            .scan(1, |a, b| {
                *a *= *b;
                Some(*a)
            })
            .collect();
        for (i, x) in self.data.iter().enumerate() {
            for y in &t {
                if i % y == 0 {
                    write!(f, "[")?;
                }
            }
            write!(f, "{:?}", x)?;
            for y in &t {
                if i % y == y - 1 {
                    write!(f, "]")?;
                }
            }
            if i != self.data.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "")
    }
}
