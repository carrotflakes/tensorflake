use std::sync::{Arc, Mutex};

use super::FunctionCall;

#[derive(Clone)]
pub(crate) struct ComputedAttrs {
    pub name: String,
    pub creator: Option<Arc<FunctionCall>>,
}

pub(crate) struct ComputedInner<T> {
    pub data: T,
    pub attrs: Mutex<ComputedAttrs>,
}

pub struct Computed<T> {
    pub(crate) inner: Arc<ComputedInner<T>>,
}

impl<T> Computed<T> {
    pub fn new(data: T) -> Self {
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

    pub fn unchained(&self) -> Self
    where
        T: Clone,
    {
        let mut attrs = self.inner.attrs.lock().unwrap().clone();
        attrs.creator = None;
        Computed {
            inner: Arc::new(ComputedInner {
                data: self.inner.data.clone(),
                attrs: Mutex::new(attrs),
            }),
        }
    }
}

impl<T> std::ops::Deref for Computed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl<T> Clone for Computed<T> {
    fn clone(&self) -> Self {
        Computed {
            inner: self.inner.clone(),
        }
    }
}

impl<T> PartialEq for Computed<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T> Eq for Computed<T> {}

impl<T> std::hash::Hash for Computed<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}
