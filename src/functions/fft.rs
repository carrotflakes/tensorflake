use std::sync::{Arc, Mutex};

use ndarray::ArrayBase;
use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::*;

/// This is a reusable instance.
pub struct Fft {
    planner: Arc<Mutex<FftPlanner<f32>>>,
}

impl Fft {
    pub fn new() -> Self {
        Fft {
            planner: Arc::new(Mutex::new(FftPlanner::new())),
        }
    }

    pub fn fft(&self, x: &ComputedNDA) -> ComputedNDA {
        let shape = x.shape();
        assert!(
            shape.len() >= 2 && shape[shape.len() - 1] == 2,
            "shape {:?} cannot fft",
            shape
        );
        let size = shape[shape.len() - 2];

        let mut it = x.iter().copied();
        let mut buf: Vec<_> = (0..shape.iter().take(shape.len() - 1).product())
            .map(|_| Complex32::new(it.next().unwrap(), it.next().unwrap()))
            .collect();

        let fft = self.planner.lock().unwrap().plan_fft_forward(size);

        for i in 0..shape.iter().take(shape.len() - 2).product() {
            fft.process(&mut buf[i * size..(i + 1) * size]);
        }

        let buf_iter = buf.iter().flat_map(|x| [x.re, x.im]);
        let y = Computed::new(ArrayBase::from_iter(buf_iter).into_shape(shape).unwrap());

        let this = self.clone();
        chain(
            &[x.clone()],
            &[y.clone()],
            false,
            "fft",
            move |_xs, _ys, gys| {
                let gx = this.ifft(&gys[0]);

                vec![gx]
            },
        );

        y
    }

    pub fn ifft(&self, x: &ComputedNDA) -> ComputedNDA {
        let shape = x.shape();
        assert!(
            shape.len() >= 2 && shape[shape.len() - 1] == 2,
            "shape {:?} cannot ifft",
            shape
        );
        let size = shape[shape.len() - 2];

        let mut it = x.iter().copied();
        let mut buf: Vec<_> = (0..shape.iter().take(shape.len() - 1).product())
            .map(|_| Complex32::new(it.next().unwrap(), it.next().unwrap()))
            .collect();

        let fft = self.planner.lock().unwrap().plan_fft_inverse(size);

        for i in 0..shape.iter().take(shape.len() - 2).product() {
            fft.process(&mut buf[i * size..(i + 1) * size]);
        }

        let buf_iter = buf.iter().flat_map(|x| [x.re, x.im]);
        let y = Computed::new(ArrayBase::from_iter(buf_iter).into_shape(shape).unwrap());

        let this = self.clone();
        chain(
            &[x.clone()],
            &[y.clone()],
            false,
            "ifft",
            move |_xs, _ys, gys| {
                let gx = this.fft(&gys[0]);

                vec![gx]
            },
        );

        y
    }
}

impl Clone for Fft {
    fn clone(&self) -> Self {
        Self {
            planner: self.planner.clone(),
        }
    }
}

#[test]
fn test() {
    fn f(x: ComputedNDA) {
        let y = Fft::new().fft(&x);
        let y = y
            .pow(2.0)
            .sum((0..x.shape().len()).collect::<Vec<_>>(), false);
        let before_y = y[[]];
        let gs = gradients(&[y], &[x.clone()], false);
        let x = &x - &(&gs[0] * &backprop(scalar(0.01)));
        let y = Fft::new().fft(&x);
        let y = y
            .pow(2.0)
            .sum((0..x.shape().len()).collect::<Vec<_>>(), false);
        let after_y = y[[]];
        assert!(after_y < before_y);
    }
    f(backprop(
        ndarray::Array2::from_shape_fn((16, 2), |(i, j)| {
            if j == 0 {
                (4.0 * i as f32 / 16.0 * std::f32::consts::TAU).sin()
            } else {
                0.0
            }
        })
        .into_ndarray(),
    ));
    f(backprop(
        ndarray::Array2::from_shape_fn((16, 2), |(i, j)| {
            if j == 0 {
                (4.0 * i as f32 / 16.0 * std::f32::consts::TAU).cos()
            } else {
                0.0
            }
        })
        .into_ndarray(),
    ));
    f(backprop(
        ndarray::Array2::from_shape_fn((16, 2), |(i, j)| {
            ((i.pow(2) as f32 + j as f32 * 32.0) / 16.0 * std::f32::consts::TAU).sin()
        })
        .into_ndarray(),
    ));
    f(backprop(
        ndarray::Array3::from_shape_fn((3, 16, 2), |(k, i, j)| {
            ((i.pow(2) as f32 + j as f32 * 32.0 + (i * k) as f32) / 16.0 * std::f32::consts::TAU)
                .sin()
        })
        .into_ndarray(),
    ));
}
