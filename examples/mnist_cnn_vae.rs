mod data;

use ndarray_rand::{rand_distr::Normal, RandomExt};
use tensorflake::{
    initializers::Initializer,
    losses::sigmoid_cross_entropy_with_logits,
    nn::{
        activations::{relu, sigmoid},
        *,
    },
    training::TrainConfig,
    *,
};

fn main() {
    let mnist = data::mnist::Mnist::load("./data/mnist");

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
            let ys = model.call(&x, ctx.train);
            let cross_ent = sigmoid_cross_entropy_with_logits(&x, &ys[0]);
            let reconstruct_loss = cross_ent.sum(vec![0, 1, 2, 3], false)
                * Tensor::new(scalar(1.0 / batch.len() as f32));
            let kl_loss =
                -log_normal_pdf(&ys[1], &Tensor::new(scalar(0.0)), &Tensor::new(scalar(0.0)))
                    + log_normal_pdf(&ys[1], &ys[2], &ys[3]);
            let loss = reconstruct_loss + kl_loss;
            // graph(&[loss.clone()], "vae");
            // panic!();
            ctx.finish_batch(&loss, batch.len());
        });

        // generate images
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
        let ys = model.call(&x, false);
        let y = sigmoid(&ys[0]);
        save_iamges(
            &functions::Concat::new(0).call(vec![x, y.clone()])[0],
            &format!("vae_image_{}.png", train.epoch),
        );
    }

    println!("time: {:?}", start.elapsed());
}

pub struct Model {
    pub encoder_convs: [Conv2d; 2],
    pub encoder_linear1: Linear,
    pub encoder_linear2: Linear,
    pub decoder_linear: Linear,
    pub decoder_convts: [Conv2dTranspose; 2],
    pub decoder_conv: Conv2d,
    // pub rng: DefaultRng,
}

impl Model {
    pub fn new() -> Self {
        let optimizer = optimizers::AdamOptimizer::new();
        let init_kernel = initializers::InitializerWithOptimizer::new(
            Normal::new(0.0, 0.1).unwrap(),
            optimizer.clone(),
        );
        let init_bias = initializers::InitializerWithOptimizer::new(
            Normal::new(0.0, 0.0).unwrap(),
            optimizer.clone(),
        );
        let latent_dim = 2;
        Self {
            encoder_convs: [
                Conv2d::new(
                    1,
                    32,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    init_kernel.scope("conv2d_0_w"),
                    Some(init_bias.scope("conv2d_0_b")),
                ),
                Conv2d::new(
                    32,
                    64,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    init_kernel.scope("conv2d_1_w"),
                    Some(init_bias.scope("conv2d_1_b")),
                ),
            ],
            encoder_linear1: Linear::new(
                64 * 7 * 7,
                latent_dim,
                init_kernel.scope("encoder_linear1_w"),
                Some(init_bias.scope("encoder_linear1_b")),
            ),
            encoder_linear2: Linear::new(
                64 * 7 * 7,
                latent_dim,
                init_kernel.scope("encoder_linear2_w"),
                Some(init_bias.scope("encoder_linear2_b")),
            ),
            decoder_linear: Linear::new(
                latent_dim,
                32 * 7 * 7,
                init_kernel.scope("decoder_linear_w"),
                Some(init_bias.scope("decoder_linear_b")),
            ),
            decoder_convts: [
                Conv2dTranspose::new(
                    64,
                    32,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    Some([14, 14]),
                    init_kernel.scope("decoder_conv2d_transpose_0_w"),
                    Some(init_bias.scope("decoder_conv2d_transpose_0_b")),
                ),
                Conv2dTranspose::new(
                    32,
                    64,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    Some([28, 28]),
                    init_kernel.scope("decoder_conv2d_transpose_1_w"),
                    Some(init_bias.scope("decoder_conv2d_transpose_1_b")),
                ),
            ],
            decoder_conv: Conv2d::new(
                32,
                1,
                [3, 3],
                [1, 1],
                [1, 1],
                init_kernel.scope("decoder_conv2d_w"),
                Some(init_bias.scope("decoder_conv2d_b")),
            ),
            // rng: DefaultRng::seed_from_u64(42),
        }
    }

    pub fn encode(&self, x: &Tensor, train: bool) -> Tensor {
        let mut x = x.clone();
        for conv in &self.encoder_convs {
            x = conv.call(x, train);
            x = relu(&x);
        }

        x = x.reshape(vec![x.shape()[0], 64 * 7 * 7]);
        x
    }

    pub fn decode(&self, x: &Tensor, train: bool) -> Tensor {
        let mut x = relu(&self.decoder_linear.call(x.clone(), train));
        x = x.reshape(vec![x.shape()[0], 32, 7, 7]);

        for conv in &self.decoder_convts {
            x = conv.call(x, train);
            x = relu(&x);
        }
        x = self.decoder_conv.call(x, train);
        x
    }

    pub fn call(&self, x: &Tensor, train: bool) -> [Tensor; 4] {
        let mut x = self.encode(&x, train);

        let mean = self.encoder_linear1.call(x.clone(), train).named("mean");
        let log_var = self.encoder_linear2.call(x.clone(), train).named("log_var");
        let noise = Tensor::new(NDArray::random(
            mean.shape(),
            Normal::new(0.0, 1.0).unwrap(),
        ));
        let z = if train {
            &mean + &(&noise * &log_var.exp())
        } else {
            mean.clone()
        }
        .named("z");

        x = self.decode(&z, train);
        // x = sigmoid(&x);
        [x, z, mean, log_var]
    }

    pub fn all_params(&self) -> Vec<Param> {
        [].into_iter()
            .chain(self.encoder_convs.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_convts.iter().flat_map(|x| x.all_params()))
            .chain(self.decoder_conv.all_params())
            .collect()
    }
}

fn log_normal_pdf(sample: &Tensor, mean: &Tensor, log_var: &Tensor) -> Tensor {
    let log2pi = (2.0 * std::f32::consts::PI).ln();
    (Tensor::new(scalar(-0.5))
        * ((sample - mean).pow(2.0) * (-log_var).exp()
            + log_var.clone()
            + Tensor::new(scalar(log2pi))))
    .sum(Vec::from_iter(0..sample.ndim()), false)
        * Tensor::new(scalar(1.0 / sample.len() as f32))
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

#[allow(dead_code)]
fn graph(vars: &[Tensor], name: impl ToString) {
    let f = std::fs::File::create(name.to_string() + ".dot").unwrap();
    let mut w = std::io::BufWriter::new(f);
    tensorflake::export_dot::write_dot(&mut w, vars, &mut |v| {
        // format!("{} {}", v.get_name(), (*v).to_string())
        // v.get_name().to_string()
        format!("{} {:?}", v.get_name(), v.shape())
    })
    .unwrap();
}
