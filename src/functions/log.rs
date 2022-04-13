use crate::*;

pub fn log(x: &Tensor) -> Tensor {
    let y = Tensor::new(x.map(|x| x.ln()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "log",
        move |xs, _, gys| vec![&gys[0] / &xs[0]],
    );

    y
}
