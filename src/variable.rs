use std::cell::RefCell;
use std::rc::Rc;

use crate::{collect_funcalls, functions::Add, sort_for_backward, Funcall, Function, Tensor};

pub(crate) struct VariableAttrs {
    grad: Option<Rc<VariableInner>>,
    pub creator: Option<Rc<Funcall>>,
    pub name: String,
}

pub(crate) struct VariableInner {
    pub data: Tensor,
    pub attrs: RefCell<VariableAttrs>,
}

pub struct Variable<const ENABLE_BACKPROP: bool = false> {
    pub(crate) inner: Rc<VariableInner>,
}

impl<const ENABLE_BACKPROP: bool> Variable<ENABLE_BACKPROP> {
    pub fn new(data: Tensor) -> Self {
        Variable {
            inner: Rc::new(VariableInner {
                data,
                attrs: RefCell::new(VariableAttrs {
                    grad: None,
                    creator: None,
                    name: String::new(),
                }),
            }),
        }
    }

    pub fn named(self, name: impl Into<String>) -> Self {
        self.inner.attrs.borrow_mut().name = name.into();
        self
    }

    pub fn set_name(&self, name: impl Into<String>) {
        self.inner.attrs.borrow_mut().name = name.into();
    }

    pub fn get_name(&self) -> String {
        self.inner.attrs.borrow().name.to_owned()
    }

    pub fn flip_bp<const EB: bool>(&self) -> Variable<EB> {
        Variable {
            inner: self.inner.clone(),
        }
    }
}

impl Variable<true> {
    pub fn get_grad<const ENABLE_BACKPROP: bool>(&self) -> Option<Variable<ENABLE_BACKPROP>> {
        self.inner
            .attrs
            .borrow()
            .grad
            .as_ref()
            .map(|i| Variable { inner: i.clone() })
    }

    pub fn set_grad<const ENABLE_BACKPROP: bool>(&self, grad: Variable<ENABLE_BACKPROP>) {
        // broadcast
        // if self.shape() != grad.shape() {
        //     grad = call!(BroadcastTo::new(self.shape().to_vec()), grad);
        // }

        self.inner.attrs.borrow_mut().grad = Some(grad.inner);
    }

    pub fn add_grad<const ENABLE_BACKPROP: bool>(&self, v: Variable<ENABLE_BACKPROP>) {
        // broadcast
        // if self.shape() != v.shape() {
        //     assert!(v.shape().iter().product::<usize>() <= self.shape().iter().product(), "invalid broadcast: {:?} to {:?} in {:?}", v.shape(), self.shape(), self.get_name());
        //     v = call!(BroadcastTo::new(self.shape().to_vec()), v);
        // }

        let grad = &mut self.inner.attrs.borrow_mut().grad;
        if let Some(grad) = grad.as_mut() {
            *grad = Add
                .call(vec![
                    Variable {
                        inner: grad.clone(),
                    },
                    v,
                ])
                .pop()
                .unwrap()
                .inner;
        } else {
            *grad = Some(v.inner);
        }
    }

    pub fn clear_grad(&self) {
        self.inner.attrs.borrow_mut().grad = None;
    }

    pub fn backward(&self, retain_grad: bool, create_graph: bool) {
        // ensure grad is not None
        self.inner.attrs.borrow_mut().grad.get_or_insert_with(|| {
            Variable::<true>::new(ndarray::Array::ones(self.inner.data.shape()).into_dyn()).inner
        });

        let funcalls = collect_funcalls(vec![self.clone()]);
        for fc in sort_for_backward(funcalls) {
            fc.backward(retain_grad, create_graph);
        }
    }
}

impl<const ENABLE_BACKPROP: bool> std::ops::Deref for Variable<ENABLE_BACKPROP> {
    type Target = Tensor;

    fn deref(&self) -> &Tensor {
        &self.inner.data
    }
}

impl<const ENABLE_BACKPROP: bool> Clone for Variable<ENABLE_BACKPROP> {
    fn clone(&self) -> Self {
        Variable {
            inner: self.inner.clone(),
        }
    }
}

impl<const ENABLE_BACKPROP: bool> PartialEq for Variable<ENABLE_BACKPROP> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}
