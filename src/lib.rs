pub mod export_dot;
mod funcall;
mod function;
pub mod functions;
pub mod nn;
mod variable;

#[cfg(test)]
mod test;

use std::rc::Rc;

pub use funcall::*;
pub use function::*;
pub use variable::*;

pub const ENABLE_BACKPROP: bool = true;
pub const DISABLE_BACKPROP: bool = false;

pub type Tensor = ndarray::ArrayD<f32>;

pub fn scalar(x: f32) -> Tensor {
    ndarray::arr0(x).into_dyn()
}

#[macro_export]
macro_rules! call {
    ($e:expr, $($es:expr),*) => {
        $e.call(vec![$($es.clone()),*]).pop().unwrap()
    };
}

pub(crate) fn collect_funcalls(mut vars: Vec<Variable<ENABLE_BACKPROP>>) -> Vec<Rc<Funcall>> {
    let mut funcall_vec = Vec::new();
    let mut closed_vars = Vec::new();
    while let Some(var) = vars.pop() {
        if closed_vars.contains(&var) {
            continue;
        }
        closed_vars.push(var.clone());

        if let Some(creator) = var.inner.attrs.borrow().creator.clone() {
            vars.extend(creator.input.iter().cloned());
            vars.extend(creator.output.iter().cloned());
            funcall_vec.push(creator);
        }
    }
    funcall_vec
}

pub fn collect_variables(vars: Vec<Variable<ENABLE_BACKPROP>>) -> Vec<Variable<ENABLE_BACKPROP>> {
    let fcs = collect_funcalls(vars);
    let mut vars: Vec<_> = fcs.iter().flat_map(|fc| fc.input.iter()).cloned().collect();
    vars.dedup();
    vars
}

pub fn release_variables(var: &Variable<ENABLE_BACKPROP>) {
    let fcs = collect_funcalls(vec![var.clone()]);
    for v in fcs.iter().flat_map(|fc| fc.input.iter().cloned()) {
        v.clear_grad();
        v.inner.attrs.borrow_mut().creator = None;
    }
}

#[test]
fn test_collect_funcalls() {
    let x = Variable::<true>::new(scalar(1.0));
    let y = Variable::new(scalar(2.0));
    let z = Variable::new(scalar(3.0));
    let f = functions::Add.call(vec![x.clone(), y.clone()]);
    let g = functions::Add.call([f.clone(), vec![z.clone()]].concat());
    let f = functions::Add.call([g.clone(), vec![x.clone()]].concat());
    let funcall_vec = collect_funcalls(vec![f[0].clone()]);
    assert_eq!(funcall_vec.len(), 3);
}
