/// COIN: COmpression with Implicit Neural representations https://arxiv.org/abs/2103.03123
use std::sync::{Arc, Mutex};

use image::GenericImageView;
use ndarray_rand::rand_distr::Uniform;
use tensorflake::{
    initializers::{Initializer, InitializerWithSharedOptimizer},
    losses::*,
    nn::*,
    training::{TrainConfig, UpdateStrategy},
    *,
};

fn main() {
    let img = image::io::Reader::open("img.png")
        .unwrap()
        .decode()
        .unwrap();

    let init = InitializerWithSharedOptimizer::new(
        Uniform::new(0., 0.01),
        Arc::new(Mutex::new(optimizers::AdamOptimizer::new())),
    );

    let model = Model::new(init.scope("coin"), init.scope("coin"));
    // param_bin::import_from_file(&mut model.all_params(), "coin_50.bin");
    // gen_image([img.height(), img.width()], &model, "coin_final.png");
    // return;

    let start = std::time::Instant::now();

    let mut train = TrainConfig {
        epoch: 100,
        train_data: (0..img.height())
            .flat_map(|y| (0..img.width()).map(move |x| (y, x)))
            .collect::<Vec<_>>(),
        batch_size: usize::MAX,
        parallel_chunk_size: 128,
        update_strategy: UpdateStrategy::Chunk(1),
        ..TrainConfig::default()
    }
    .build();

    while !train.is_end() {
        train.fit_one_epoch(|batch, ctx| {
            let x: Vec<_> = batch
                .iter()
                .flat_map(|(y, x)| {
                    [
                        *y as f32 / (img.height() - 1) as f32,
                        *x as f32 / (img.width() - 1) as f32,
                    ]
                })
                .collect();
            let t = batch
                .iter()
                .flat_map(|(y, x)| {
                    img.get_pixel(*x, *y).0[0..3]
                        .iter()
                        .map(|x| *x as f32 / 255.0)
                        .collect::<Vec<_>>()
                })
                .collect();
            let x = NDArray::from_shape_vec(&[batch.len(), 2][..], x).unwrap();
            let t = NDArray::from_shape_vec(&[batch.len(), 3][..], t).unwrap();
            let x = Computed::new(x.clone());
            let t = Computed::new(t.clone());

            let y = model.call(x.clone(), ctx.train);
            let loss = naive_mean_squared_error(t.clone(), y.clone());

            ctx.finish_batch(&loss, batch.len());
        });

        if train.epoch % 10 == 0 {
            gen_image(
                [img.height(), img.width()],
                &model,
                &format!("coin_{}.png", train.epoch),
            );
            param_bin::export_to_file(&model.all_params(), &format!("coin_{}.bin", train.epoch));
        }
    }

    println!("time: {:?}", start.elapsed());

    gen_image([img.height(), img.width()], &model, "coin_final.png");
}

pub struct Model {
    pub mlp: MLP,
    pub activation: Box<dyn Fn(Computed) -> Computed + Sync + Send>,
}

impl Model {
    pub fn new(w: impl Initializer, b: impl Initializer) -> Self {
        Self {
            mlp: MLP::new(
                &[2, 28, 28, 28, 28, 28, 28, 28, 28, 28, 3],
                // Some(Dropout::new(0.2, 42)),
                None,
                |x| x.sin(),
                w.scope("mlp_w"),
                b.scope("mlp_b"),
            ),
            activation: Box::new(|x| activations::sigmoid(&x)),
        }
    }
}

impl Layer for Model {
    type Input = Computed;
    type Output = Computed;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output {
        let y = self.mlp.call(x, train);
        (self.activation)(y)
    }

    fn all_params(&self) -> Vec<Param> {
        self.mlp.all_params()
    }
}

fn gen_image(size: [u32; 2], layer: &impl Layer<Input = Computed, Output = Computed>, path: &str) {
    let mut img = image::ImageBuffer::new(size[1], size[0]);
    let ps = (0..size[0])
        .flat_map(|y| (0..size[1]).map(move |x| (y, x)))
        .collect::<Vec<_>>();
    let chunk_size = 128;
    for chunk in ps.chunks(chunk_size) {
        let a = chunk
            .iter()
            .flat_map(|(y, x)| {
                [
                    *y as f32 / (size[0] - 1) as f32,
                    *x as f32 / (size[1] - 1) as f32,
                ]
            })
            .collect::<Vec<_>>();
        let b = layer.call(
            NDArray::from_shape_vec(&[chunk.len(), 2][..], a)
                .unwrap()
                .into(),
            false,
        );
        for (i, (y, x)) in chunk.iter().enumerate() {
            img.put_pixel(
                *x,
                *y,
                image::Rgb([
                    (b[[i, 0]] * 255.0) as u8,
                    (b[[i, 1]] * 255.0) as u8,
                    (b[[i, 2]] * 255.0) as u8,
                ]),
            );
        }
    }
    img.save(path).unwrap();
    println!("saved {}", path);
}
