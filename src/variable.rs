use std::sync::{Arc, Mutex};

use crate::{Funcall, Tensor};

pub(crate) struct VariableAttrs {
    pub name: String,
    pub trainable: bool,
    // pub recorder: Option<Recorder>, // todo: weak
    pub creator: Option<Arc<Funcall>>,
}

pub(crate) struct VariableInner {
    pub data: Tensor,
    pub attrs: Mutex<VariableAttrs>,
}

pub struct Variable {
    pub(crate) inner: Arc<VariableInner>,
}

impl Variable {
    pub fn new(data: Tensor) -> Self {
        Variable {
            inner: Arc::new(VariableInner {
                data,
                attrs: Mutex::new(VariableAttrs {
                    name: "".to_string(),
                    trainable: true,
                    // recorder: None,
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

    // pub fn recorded(self) -> Self {
    //     self.inner
    //         .attrs
    //         .lock()
    //         .unwrap()
    //         .recorder
    //         .get_or_insert(Recorder::new());
    //     self
    // }

    // pub fn set_recorder(&self, recorder: Recorder) {
    //     self.inner.attrs.lock().unwrap().recorder = Some(recorder);
    // }

    pub fn has_creator(&self) -> bool {
        self.inner.attrs.lock().unwrap().creator.is_some()
    }
}

impl std::ops::Deref for Variable {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner.data
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        Variable {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Variable {}

impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}
