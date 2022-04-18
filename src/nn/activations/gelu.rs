use crate::*;

pub fn gelu(x: &Tensor) -> Tensor {
    // x * &(((x + &(Tensor::new(scalar(0.044715)) * x.pow(3.0)))
    //     * Tensor::new(scalar((2.0 / std::f32::consts::PI).sqrt())))
    // .tanh()
    //     + Tensor::new(scalar(1.0)))
    x * &super::sigmoid(&(x * &Tensor::new(scalar(1.702))))
}

#[test]
fn test() {
    let x = Tensor::new(NDArray::from_shape_vec(&[4][..], vec![-2.0, -1.0, 0.0, 1.0]).unwrap());
    let y = gelu(&x);
    assert_eq!(y[2], 0.0);
    assert!(y[0] > y[1]);
    assert!(y[0] < y[2]);
    assert!(y[1] < y[2]);
    assert!(y[2] < y[3]);
}
