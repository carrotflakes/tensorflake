use std::sync::{Arc, Mutex};

use super::{FunctionCall, NDArray};

#[derive(Clone)]
pub(crate) struct ComputedAttrs {
    pub name: String,
    pub creator: Option<Arc<FunctionCall>>,
}

pub(crate) struct ComputedInner {
    pub data: NDArray,
    pub attrs: Mutex<ComputedAttrs>,
}

pub struct Computed {
    pub(crate) inner: Arc<ComputedInner>,
}

impl Computed {
    pub fn new(data: NDArray) -> Self {
        Computed {
            inner: Arc::new(ComputedInner {
                data,
                attrs: Mutex::new(ComputedAttrs {
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

    pub fn unchain(&self) {
        self.inner.attrs.lock().unwrap().creator = None;
    }
}

impl std::ops::Deref for Computed {
    type Target = NDArray;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl Clone for Computed {
    fn clone(&self) -> Self {
        Computed {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for Computed {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Computed {}

impl std::hash::Hash for Computed {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl Into<Computed> for NDArray {
    fn into(self) -> Computed {
        Computed::new(self)
    }
}
