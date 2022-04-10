mod data;

use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use tensorflake::{
    losses::naive_mean_squared_error,
    nn::{
        activations::{Relu, Sigmoid},
        *,
    },
    training::TrainConfig,
    *,
};

fn main() {
    let mnist = data::mnist::Mnist::load("./data/fashion-mnist");

    let model = Model::new();
    // param_bin::params_summary(&model.all_params());

    let start = std::time::Instant::now();

    let mut train = TrainConfig {
        epoch: 10,
        train_data: mnist.trains().collect(),
        validation_data: mnist.tests().collect(),
        validation_rate: 0.1,
        batch_size: 32,
        parallel: false,
        ..Default::default()
    }
    .build();
    while !train.is_end() {
        train.fit_one_epoch(|batch, ctx| {
            let x = Tensor::new(
                NDArray::from_shape_vec(
                    &[batch.len(), 1, 28, 28][..],
                    batch
                        .iter()
                        .flat_map(|x| x.0)
                        .map(|x| *x as f32 / 255.0)
                        .collect(),
                )
                .unwrap(),
            );
            let y = model.call(x.clone(), ctx.train);
            let loss = naive_mean_squared_error(x.clone(), y.clone());
            if ctx.train {
                optimize(&loss);
            }
            ctx.count(batch.len());
            ctx.add_metric(metrics::Loss::new(loss[[]], batch.len()));
        });

        let x = Tensor::new(
            NDArray::from_shape_vec(
                &[32, 1, 28, 28][..],
                mnist
                    .trains()
                    .take(32)
                    .flat_map(|x| x.0)
                    .map(|x| *x as f32 / 255.0)
                    .collect(),
            )
            .unwrap(),
        );
        let y = model.call(x.clone(), false);
        save_iamges(
            &functions::Concat::new(0).call(vec![x, y])[0],
            &format!("ae_image_{}.png", train.epoch),
        );
    }

    println!("time: {:?}", start.elapsed());
}

pub struct Model {
    pub encoder_convs: [Conv2d; 2],
    pub decoder_convts: [Conv2dTranspose; 2],
    pub decoder_conv: Conv2d,
}

impl Model {
    pub fn new() -> Self {
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
        let param_gen = {
            rng.gen::<u32>();
            let rng = rng.clone();
            move || {
                let mut rng = rng.clone();
                move |shape: &[usize]| -> Param {
                    let t = NDArray::random_using(shape, Uniform::new(-0.01, 0.01), &mut rng);
                    Param::new(t, optimizers::AdamOptimizer::new())
                }
            }
        };
        Self {
            encoder_convs: [
                Conv2d::new(
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    param_gen()(&[16, 1, 3, 3]),
                    Some(param_gen()(&[16])),
                ),
                Conv2d::new(
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    param_gen()(&[8, 16, 3, 3]),
                    Some(param_gen()(&[8])),
                ),
            ],
            decoder_convts: [
                Conv2dTranspose::new(
                    [2, 2],
                    [1, 1],
                    [14, 14],
                    param_gen()(&[8, 8, 3, 3]),
                    Some(param_gen()(&[8])),
                ),
                Conv2dTranspose::new(
                    [2, 2],
                    [1, 1],
                    [28, 28],
                    param_gen()(&[8, 16, 3, 3]),
                    Some(param_gen()(&[16])),
                ),
            ],
            decoder_conv: Conv2d::new(
                [3, 3],
                [1, 1],
                [1, 1],
                param_gen()(&[1, 16, 3, 3]),
                Some(param_gen()(&[1])),
            ),
        }
    }

    pub fn call(&self, x: Tensor, train: bool) -> Tensor {
        let mut x = x;
        for conv in &self.encoder_convs {
            x = conv.call(x, train);
            x = Relu.call(vec![x]).pop().unwrap();
        }
        for conv in &self.decoder_convts {
            x = conv.call(x, train);
            x = Relu.call(vec![x]).pop().unwrap();
        }
        x = self.decoder_conv.call(x, train);
        x = Sigmoid.call(vec![x]).pop().unwrap();
        x
    }

    pub fn all_params(&self) -> Vec<Param> {
        [].into_iter()
            .chain(self.encoder_convs.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_convts.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_conv.all_params())
            .collect()
    }
}

fn save_iamges(data: &Tensor, path: &str) {
    let mut img = image::ImageBuffer::new(data.shape()[3] as u32 * 8, data.shape()[2] as u32 * 8);

    for i in 0..data.shape()[0] {
        for y in 0..data.shape()[2] {
            for x in 0..data.shape()[3] {
                let v = data[[i, 0, y, x]];
                let color = image::Rgb([(v * 255.0) as u8, (v * 255.0) as u8, (v * 255.0) as u8]);
                img.put_pixel(
                    ((i % 8) * data.shape()[3] + x) as u32,
                    ((i / 8) * data.shape()[2] + y) as u32,
                    color,
                );
            }
        }
    }

    img.save(path).unwrap();
    println!("saved {}", path);
}
