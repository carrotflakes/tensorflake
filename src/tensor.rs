use std::sync::{Arc, Mutex};

use crate::{FunctionCall, NDArray};

#[derive(Clone)]
pub(crate) struct TensorAttrs {
    pub name: String,
    pub creator: Option<Arc<FunctionCall>>,
}

pub(crate) struct TensorInner {
    pub data: NDArray,
    pub attrs: Mutex<TensorAttrs>,
}

pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

impl Tensor {
    pub fn new(data: NDArray) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                data,
                attrs: Mutex::new(TensorAttrs {
                    name: "".to_string(),
                    creator: None,
                }),
            }),
        }
    }

    pub fn named(self, name: impl Into<String>) -> Self {
        self.inner.attrs.lock().unwrap().name = name.into();
        self
    }

    pub fn set_name(&self, name: impl Into<String>) {
        self.inner.attrs.lock().unwrap().name = name.into();
    }

    pub fn get_name(&self) -> String {
        self.inner.attrs.lock().unwrap().name.to_owned()
    }

    pub fn has_creator(&self) -> bool {
        self.inner.attrs.lock().unwrap().creator.is_some()
    }

    pub fn cut_chain(&self) {
        self.inner.attrs.lock().unwrap().creator = None;
    }
}

impl std::ops::Deref for Tensor {
    type Target = NDArray;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Tensor {}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl Into<Tensor> for NDArray {
    fn into(self) -> Tensor {
        Tensor::new(self)
    }
}
