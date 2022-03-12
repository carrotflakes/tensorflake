use std;

use std::rc::Rc;

use std::cell::RefCell;

use crate::Funcall;

pub struct VariableInner {
    pub(crate) data: f64,
    pub(crate) grad: RefCell<Option<Variable>>,
    pub(crate) creator: RefCell<Option<Rc<Funcall>>>,
}

pub struct Variable {
    pub(crate) inner: Rc<VariableInner>,
}

impl Variable {
    pub fn new(data: f64) -> Variable {
        Variable {
            inner: Rc::new(VariableInner {
                data,
                grad: RefCell::new(None),
                creator: RefCell::new(None),
            }),
        }
    }

    pub fn get_grad(&self) -> Option<Variable> {
        self.inner.grad.borrow().clone()
    }

    pub fn set_grad(&self, grad: Variable) {
        *self.inner.grad.borrow_mut() = Some(grad);
    }

    pub fn backward(&self) {
        if let Some(creator) = self.inner.creator.borrow().clone() {
            creator.backward();
            for v in &creator.input {
                v.backward();
            }
        }
    }
}

impl std::ops::Deref for Variable {
    type Target = f64;

    fn deref(&self) -> &f64 {
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
