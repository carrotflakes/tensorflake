use crate::{functions::*, tensor_util::as_2d, *};

use ndarray::s;
use ndarray_rand::rand::Rng;

use super::{Linear, Softmax};

pub struct SelectNet {
    pub output_size: usize,
    pub select_layer: Linear,
    pub layers: Vec<Linear>,
}

impl SelectNet {
    pub fn new(
        input: usize,
        output: usize,
        n: usize,
        param_gen: &(impl Fn(Tensor) -> Box<dyn Fn() -> Variable> + 'static),
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            output_size: output,
            select_layer: Linear::new(input, n, param_gen, rng),
            layers: (0..n)
                .map(|_| Linear::new(input, output, param_gen, rng))
                .collect(),
        }
    }

    pub fn build(&self) -> Self {
        let select_layer = self.select_layer.build();
        let layers = self.layers.iter().map(|layer| layer.build()).collect();
        Self {
            output_size: self.output_size,
            select_layer,
            layers,
        }
    }

    pub fn call(&self, xs: Vec<Variable>) -> Vec<Variable> {
        let select = self.select_layer.call(xs.to_vec()).pop().unwrap();
        let x = as_2d(&xs[0]);
        let softmax = call!(Softmax, select);

        let mut ys = vec![];
        for i in 0..x.shape()[0] {
            let mut select = (&*softmax)
                .slice(s![i, ..].to_owned())
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            select.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            select.truncate(5);

            let mut lys = Vec::new();
            for (j, _) in &select {
                let layer = &self.layers[*j];
                let ly = layer
                    .call(vec![Variable::new(x.slice(s![i..=i, ..]).into_tensor())])
                    .pop()
                    .unwrap();
                lys.push(call!(Mul, ly, call!(Slice::new(s![i, *j]), softmax)));
            }
            let lys = Add.call(lys).pop().unwrap();
            ys.push(lys);
        }

        let y = Concat::new(1).call(ys).pop().unwrap();
        // reshape to original shape
        let y = call!(
            Reshape::new(
                xs[0]
                    .shape()
                    .iter()
                    .take(xs[0].ndim() - 1)
                    .chain([self.output_size].iter())
                    .copied()
                    .collect::<Vec<_>>()
            ),
            y
        );

        vec![y, softmax]
    }
}

#[test]
fn test() {
    use ndarray::array;
    use ndarray_rand::rand::SeedableRng;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);

    let select_net = SelectNet::new(
        2,
        3,
        10,
        &|x| {
            let o = crate::optimizees::MomentumSGDOptimizee::new(x, 0.9);
            Box::new(move || o.get())
        },
        &mut rng,
    );

    let x = backprop(array![[0.1, 0.2], [0.0, 0.0], [0.0, 100.0]].into_tensor());
    let y = select_net.build().call(vec![x.clone()]);
    dbg!(&*y[0]);
    // dbg!(&*y[1]);
    let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_tensor()
        .into();
    let loss = losses::naive_mean_squared_error(y[0].clone(), t);
    dbg!(loss[[]]);
    // export_dot::export_dot(&[loss.clone()], &format!("select_net.dot")).unwrap();
    optimize(&loss, 0.1);

    let y = select_net.build().call(vec![x.clone()]);
    dbg!(&*y[0]);
}
