use std;

use std::rc::Rc;

use std::cell::RefCell;

use crate::{collect_funcalls, functions::Add, Funcall, Function, Tensor};

pub(crate) struct VariableInner {
    pub data: Tensor,
    grad: RefCell<Option<Rc<VariableInner>>>,
    pub creator: RefCell<Option<Rc<Funcall>>>,
    pub generation: u32,
}

pub struct Variable<const ENABLE_BACKPROP: bool = false> {
    pub(crate) inner: Rc<VariableInner>,
}

impl<const ENABLE_BACKPROP: bool> Variable<ENABLE_BACKPROP> {
    pub fn new(data: Tensor) -> Self {
        Variable {
            inner: Rc::new(VariableInner {
                data,
                grad: RefCell::new(None),
                creator: RefCell::new(None),
                generation: 0,
            }),
        }
    }
}

impl Variable<true> {
    pub fn new_with_gen(data: Tensor, generation: u32) -> Self {
        Variable {
            inner: Rc::new(VariableInner {
                data,
                grad: RefCell::new(None),
                creator: RefCell::new(None),
                generation,
            }),
        }
    }

    pub fn get_next_gen(vars: &[Variable<true>]) -> u32 {
        vars.iter().map(|v| v.inner.generation).max().unwrap_or(0) + 1
    }

    pub fn get_grad<const ENABLE_BACKPROP: bool>(&self) -> Option<Variable<ENABLE_BACKPROP>> {
        self.inner
            .grad
            .borrow()
            .as_ref()
            .map(|i| Variable { inner: i.clone() })
    }

    pub fn set_grad<const ENABLE_BACKPROP: bool>(&self, grad: Variable<ENABLE_BACKPROP>) {
        assert_eq!(self.shape, grad.shape);
        *self.inner.grad.borrow_mut() = Some(grad.inner);
    }

    pub fn add_grad<const ENABLE_BACKPROP: bool>(&self, v: Variable<ENABLE_BACKPROP>) {
        let mut grad = self.inner.grad.borrow_mut();
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
        *self.inner.grad.borrow_mut() = None;
    }

    pub fn backward(&self, retain_grad: bool, create_graph: bool) {
        let mut funcalls = collect_funcalls(vec![self.clone()]);
        funcalls.sort_by_key(|fc| -(fc.generation as i32));
        for fc in funcalls {
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
