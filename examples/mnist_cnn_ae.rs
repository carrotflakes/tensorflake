mod data;

use ndarray_rand::rand_distr::Normal;
use tensorflake::{
    initializers::{
        random_initializer::RandomInitializer, with_optimizer::InitializerWithOptimizer, Scope,
    },
    losses::naive_mean_squared_error,
    nn::{
        activations::{relu, sigmoid},
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
        ..Default::default()
    }
    .build();
    while !train.is_end() {
        train.fit_one_epoch(|batch, ctx| {
            let x = ComputedNDA::new(
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
            ctx.finish_batch(&loss, batch.len());
        });

        // generate images
        let x = ComputedNDA::new(
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
            &functions::concat(&[x, y], 0),
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
        let optimizer = optimizers::Adam::new();
        let init_kernel = InitializerWithOptimizer::new(
            RandomInitializer::new(Normal::new(0.0, 0.1).unwrap()),
            optimizer.clone(),
        );
        let init_bias = InitializerWithOptimizer::new(
            RandomInitializer::new(Normal::new(0.0, 0.0).unwrap()),
            optimizer.clone(),
        );
        Self {
            encoder_convs: [
                Conv2d::new(
                    1,
                    16,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    init_kernel.scope("conv2d_0_w"),
                    Some(init_bias.scope("conv2d_0_b")),
                ),
                Conv2d::new(
                    16,
                    8,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    init_kernel.scope("conv2d_1_w"),
                    Some(init_bias.scope("conv2d_1_b")),
                ),
            ],
            decoder_convts: [
                Conv2dTranspose::new(
                    8,
                    8,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    Some([14, 14]),
                    init_kernel.scope("conv2d_transpose_0_w"),
                    Some(init_bias.scope("conv2d_transpose_0_b")),
                ),
                Conv2dTranspose::new(
                    16,
                    8,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    Some([28, 28]),
                    init_kernel.scope("conv2d_transpose_1_w"),
                    Some(init_bias.scope("conv2d_transpose_1_b")),
                ),
            ],
            decoder_conv: Conv2d::new(
                16,
                1,
                [3, 3],
                [1, 1],
                [1, 1],
                init_kernel.scope("decoder_conv_w"),
                Some(init_bias.scope("decoder_conv_b")),
            ),
        }
    }

    pub fn call(&self, x: ComputedNDA, train: bool) -> ComputedNDA {
        let mut x = x;
        for conv in &self.encoder_convs {
            x = conv.call(x, train);
            x = relu(&x);
        }
        for conv in &self.decoder_convts {
            x = conv.call(x, train);
            x = relu(&x);
        }
        x = self.decoder_conv.call(x, train);
        x = sigmoid(&x);
        x
    }

    pub fn all_params(&self) -> Vec<ParamNDA> {
        [].into_iter()
            .chain(self.encoder_convs.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_convts.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_conv.all_params())
            .collect()
    }
}

fn save_iamges(data: &ComputedNDA, path: &str) {
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
