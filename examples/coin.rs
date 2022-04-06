use std::sync::{Arc, Mutex};

use image::{DynamicImage, GenericImageView};
use ndarray::prelude::*;
use ndarray_rand::{
    rand::{Rng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};
use tensorflake::{functions::*, losses::*, nn::*, *};

fn main() {
    let use_parallel = true;

    let img = image::io::Reader::open("img.png")
        .unwrap()
        .decode()
        .unwrap();

    let optimizer = Arc::new(Mutex::new(optimizers::AdamOptimizer::new()));
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(42);
    let param_gen = {
        rng.gen::<u32>();
        let rng = rng.clone();
        let optimizer = optimizer.clone();
        move || {
            let mut rng = rng.clone();
            let optimizer = optimizer.clone();
            move |shape: &[usize]| -> Param {
                let t = Array::random_using(shape, Uniform::new(0., 0.01), &mut rng).into_ndarray();
                Param::new_shared(t, optimizer.clone())
            }
        }
    };

    let model = Model::new(&mut param_gen(), &mut param_gen());
    // param_bin::import_from_file(&mut model.all_params(), "coin_50.bin");
    // gen_image([img.height(), img.width()], &model, "coin_final.png");
    // return;

    let batch_size = 128;

    let start = std::time::Instant::now();

    for mut ctx in ExecutionContextIter::new(100, Some((img.width() * img.height()) as usize)) {
        if !ctx.train {
            continue;
        }
        optimizer.lock().unwrap().learning_rate = 0.002 * 0.95f32.powi(ctx.epoch as i32); // MomentumSGD: 0.1, Adam: 0.001
        if use_parallel {
            use rayon::prelude::*;
            let batches: Vec<_> = mini_batches(&img, batch_size, &mut rng);
            let metrics = batches
                .par_iter()
                .map(|(len, x, t)| {
                    let x = Tensor::new(x.clone());
                    let t = Tensor::new(t.clone());
                    let y = model.call(x.clone(), ctx.train);
                    let loss = naive_mean_squared_error(t.clone(), y.clone());
                    if ctx.train {
                        optimize(&loss);
                    }
                    let mut metrics = Metrics::new();
                    metrics.count(*len);
                    metrics.add(metrics::Loss::new(loss[[]], *len));
                    metrics
                })
                .reduce(
                    || Metrics::new(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                );
            ctx.merge_metrics(metrics);
        } else {
            for (len, x, t) in mini_batches(&img, batch_size, &mut rng) {
                let x = Tensor::new(x);
                let t = Tensor::new(t);
                let y = model.call(x.clone(), ctx.train);
                let loss = naive_mean_squared_error(t.clone(), y.clone());
                if ctx.train {
                    optimize(&loss);
                }
                ctx.count(len);
                ctx.add_metric(metrics::Loss::new(loss[[]], len));
                ctx.print_progress();
            }
        }

        ctx.print_result();
        if ctx.epoch % 10 == 0 {
            gen_image(
                [img.height(), img.width()],
                &model,
                &format!("coin_{}.png", ctx.epoch),
            );
            param_bin::export_to_file(&model.all_params(), &format!("coin_{}.bin", ctx.epoch));
        }
    }

    println!("time: {:?}", start.elapsed());

    gen_image([img.height(), img.width()], &model, "coin_final.png");
}

pub struct Model {
    pub mlp: MLP,
    pub activation: Box<dyn Fn(Tensor) -> Tensor + Sync + Send>,
}

impl Model {
    pub fn new(
        w: &mut impl FnMut(&[usize]) -> Param,
        b: &mut impl FnMut(&[usize]) -> Param,
    ) -> Self {
        Self {
            mlp: MLP::new(
                &[2, 28, 28, 28, 28, 28, 28, 28, 28, 28, 3],
                // Some(Dropout::new(0.2, 42)),
                None,
                |x| call!(Sin, x),
                w,
                b,
            ),
            activation: Box::new(|x| call!(activations::Sigmoid, x)),
        }
    }
}

impl Layer for Model {
    type Input = Tensor;
    type Output = Tensor;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output {
        let y = self.mlp.call(x, train);
        (self.activation)(y)
    }

    fn all_params(&self) -> Vec<Param> {
        self.mlp.all_params()
    }
}

fn mini_batches(
    img: &DynamicImage,
    batch_size: usize,
    rng: &mut rand_isaac::Isaac64Rng,
) -> Vec<(usize, NDArray, NDArray)> {
    let p: Vec<_> = (0..img.height() * img.width())
        .map(|_| {
            (
                rng.gen_range(0..img.height()),
                rng.gen_range(0..img.width()),
            )
        })
        .collect();
    p.chunks(batch_size)
        .map(|p| {
            let x = p
                .iter()
                .flat_map(|(y, x)| {
                    [
                        *y as f32 / (img.height() - 1) as f32,
                        *x as f32 / (img.width() - 1) as f32,
                    ]
                })
                .collect();
            let y = p
                .iter()
                .flat_map(|(y, x)| {
                    img.get_pixel(*x, *y).0[0..3]
                        .iter()
                        .map(|x| *x as f32 / 255.0)
                        .collect::<Vec<_>>()
                })
                .collect();
            (
                p.len(),
                NDArray::from_shape_vec(&[p.len(), 2][..], x).unwrap(),
                NDArray::from_shape_vec(&[p.len(), 3][..], y).unwrap(),
            )
        })
        .collect()
}

fn gen_image(size: [u32; 2], layer: &impl Layer<Input = Tensor, Output = Tensor>, path: &str) {
    let mut img = image::ImageBuffer::new(size[0], size[1]);
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
