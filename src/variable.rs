use std;

use std::rc::Rc;

use std::cell::RefCell;

use crate::{collect_funcalls, Funcall};

pub(crate) struct VariableInner {
    pub data: f64,
    pub grad: RefCell<Option<Variable>>,
    pub creator: RefCell<Option<Rc<Funcall>>>,
    pub generation: u32,
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
                generation: 0,
            }),
        }
    }

    pub fn new_with_gen(data: f64, generation: u32) -> Variable {
        Variable {
            inner: Rc::new(VariableInner {
                data,
                grad: RefCell::new(None),
                creator: RefCell::new(None),
                generation,
            }),
        }
    }

    pub fn get_grad(&self) -> Option<Variable> {
        self.inner.grad.borrow().clone()
    }

    pub fn set_grad(&self, grad: Variable) {
        *self.inner.grad.borrow_mut() = Some(grad);
    }

    pub fn add_grad(&self, v: Variable) {
        let mut grad = self.inner.grad.borrow_mut();
        if let Some(g) = grad.as_mut() {
            *g = Variable::new(g.inner.data + v.inner.data);
        } else {
            *grad = Some(v);
        }
    }

    pub fn backward(&self) {
        let mut funcalls = collect_funcalls(vec![self.clone()]);
        funcalls.sort_by_key(|fc| -(fc.generation as i32));
        for fc in funcalls {
            fc.backward();
        }
    }

    pub fn get_next_gen(vars: &[Variable]) -> u32 {
        vars.iter().map(|v| v.inner.generation).max().unwrap_or(0) + 1
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

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}
