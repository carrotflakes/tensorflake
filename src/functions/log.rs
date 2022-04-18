use crate::*;

pub fn log(x: &Computed) -> Computed {
    let y = Computed::new(x.map(|x| x.ln()).into_ndarray());

    chain(
        &[x.clone()],
        &[y.clone()],
        false,
        "log",
        move |xs, _, gys| vec![&gys[0] / &xs[0]],
    );

    y
}
