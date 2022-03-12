use crate::{Variable, functions::{Sum, Mul}, Function};

#[test]
fn test_add_mul() {
    let a = Variable::new(3.0.into());
    let b = Variable::new(2.0.into());
    let c = Variable::new(1.0.into());

    let ys = Sum.call(vec![Mul.call(vec![a.clone(), b.clone()]).pop().unwrap(), c.clone()]);
    assert_eq!(*ys[0], 7.0.into());

    ys[0].set_grad(Variable::new(1.0.into()));
    ys[0].backward();
    assert_eq!(a.get_grad().map(|v| (*v).clone()), Some(2.0.into()));
    assert_eq!(b.get_grad().map(|v| (*v).clone()), Some(3.0.into()));
}
