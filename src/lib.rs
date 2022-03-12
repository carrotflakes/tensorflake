mod funcall;
mod function;
pub mod functions;
mod variable;

use std::rc::Rc;

pub use funcall::*;
pub use function::*;
pub use variable::*;

pub(crate) fn collect_funcalls(mut vars: Vec<Variable>) -> Vec<Rc<Funcall>> {
    let mut funcall_vec = Vec::new();
    let mut closed_vars = Vec::new();
    while let Some(var) = vars.pop() {
        if closed_vars.contains(&var) {
            continue;
        }
        closed_vars.push(var.clone());

        if let Some(creator) = var.inner.creator.borrow().clone() {
            vars.extend(creator.input.iter().cloned());
            vars.extend(creator.output.iter().cloned());
            funcall_vec.push(creator);
        }
    }
    funcall_vec
}

pub fn release_variables(var: &Variable) {
    let fcs = collect_funcalls(vec![var.clone()]);
    for v in fcs.iter().flat_map(|fc| fc.input.iter().cloned()) {
        v.inner.grad.replace(None);
        v.inner.creator.replace(None);
    }
}

#[test]
fn test_collect_funcalls() {
    let x = Variable::new(1.0);
    let y = Variable::new(2.0);
    let z = Variable::new(3.0);
    let f = functions::Sum.call(vec![x.clone(), y.clone()]);
    let g = functions::Sum.call([f.clone(), vec![z.clone()]].concat());
    let f = functions::Sum.call([g.clone(), vec![x.clone()]].concat());
    let funcall_vec = collect_funcalls(vec![f[0].clone()]);
    assert_eq!(funcall_vec.len(), 3);
}

// pub fn numerical_diff(f: impl Function + Sized + 'static, x: &Variable) -> Variable {
//     let eps = 1e-4;
//     let y1 = Box::new(f).call(vec![Variable::new(**x - eps)]);
//     let y2 = Box::new(f).call(vec![Variable::new(**x + eps)]);
//     Variable::new((*y2[0] - *y1[0]) / (2.0 * eps))
// }

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
