use std::{
    borrow::Cow,
    ops::AddAssign,
    sync::{Arc, Mutex},
};

use ndarray::{s, Axis, Ix1, Ix2};

use crate::{initializers::Initializer, optimizers::Fixed, param::ParamInnerT, *};

pub struct PuningLinear {
    pub output_size: usize,
    pub w: Param,
    pub b: Param,
    pub pruned_w: Arc<Mutex<Arc<Vec<(usize, usize, f32)>>>>,
}

impl PuningLinear {
    pub fn new(
        input: usize,
        output: usize,
        w: &mut impl Initializer,
        b: &mut impl Initializer,
    ) -> Self {
        let w = w.initialize(&[input, output]);
        let pruned_w = Arc::new(Mutex::new(Arc::new(prune(&w.get_tensor()))));
        let w = Param::from_inner(ParamInnerShared {
            param: w,
            pruned_w: pruned_w.clone(),
        });
        Self {
            output_size: output,
            w,
            b: b.initialize(&[output]),
            pruned_w,
        }
    }

    pub fn build(&self) -> Self {
        Self {
            output_size: self.output_size,
            w: Param::new(
                (*self.w.get_tensor()).clone(),
                self.w.get_function_name(),
                Fixed,
            ),
            b: Param::new(
                (*self.b.get_tensor()).clone(),
                self.w.get_function_name(),
                Fixed,
            ),
            pruned_w: Arc::new(Mutex::new(self.pruned_w.lock().unwrap().clone())),
        }
    }
}

impl Layer for PuningLinear {
    type Input = Computed;
    type Output = Computed;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        pruning_linear_forward(
            &x,
            &self.w.get_tensor(),
            &self.b.get_tensor(),
            self.pruned_w.lock().unwrap().clone(),
        )
    }

    fn all_params(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }
}

pub fn pruning_linear_forward(
    x: &Computed,
    w: &Computed,
    b: &Computed,
    pruned_w: Arc<Vec<(usize, usize, f32)>>,
) -> Computed {
    let xa = (**x).to_owned().into_dimensionality::<Ix2>().unwrap();
    let mut y = (**b)
        .to_owned()
        .into_dimensionality::<Ix1>()
        .unwrap()
        .broadcast([xa.shape()[0], b.len()])
        .unwrap()
        .to_owned();
    for (i, j, v) in pruned_w.iter().copied() {
        y.slice_mut(s![.., j])
            .add_assign(&(xa.slice(s![.., i]).to_owned() * v));
    }
    let y = Computed::new(y.into_ndarray());

    chain(
        &[x.clone(), w.clone()],
        &[y.clone()],
        false,
        "puning_linear",
        move |xs, _ys, gys| {
            let x = &xs[0];
            let w = &xs[1];
            let gx = gys[0].matmul(&w.mat_t()); // Producing dense matrix
            let gw = x.mat_t().matmul(&gys[0]);
            vec![gx.into(), gw]
        },
    );

    y
}

struct ParamInnerShared {
    param: Param,
    pruned_w: Arc<Mutex<Arc<Vec<(usize, usize, f32)>>>>,
}

impl ParamInnerT for ParamInnerShared {
    fn tensor(&self) -> Computed {
        self.param.get_tensor()
    }

    fn set(&mut self, tensor: Computed) {
        self.param.set((*tensor).clone());
    }

    fn update(&mut self, grad: &NDArray) {
        self.param.update(grad);

        let pruned_w = prune(&self.tensor());
        *self.pruned_w.lock().unwrap() = Arc::new(pruned_w);
    }

    fn name(&self) -> Cow<'static, str> {
        self.param.get_function_name()
    }
}

pub fn prune(x: &Computed) -> Vec<(usize, usize, f32)> {
    x.axis_iter(Axis(0))
        .enumerate()
        .flat_map(|a| {
            a.1.iter()
                .copied()
                .enumerate()
                .filter(|x| x.1.abs() > 0.0001)
                .map(|x| (a.0, x.0, x.1))
                .collect::<Vec<_>>()
        })
        .collect()
}
