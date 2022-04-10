use std::sync::{Arc, Mutex};

use crate::*;
use ndarray_rand::rand::prelude::*;
use rayon::prelude::*;

#[derive(Clone)]
pub struct TrainConfig<T> {
    pub epoch: usize,
    pub initial_epoch: usize,
    pub train_data: Vec<T>,
    pub validation_data: Vec<T>,
    pub validation_rate: f32,
    pub batch_size: usize,
    pub shuffle: bool,
    pub parallel: bool,
    pub update_strategy: UpdateStrategy,
    pub update_async: bool,
}

impl<T> Default for TrainConfig<T> {
    fn default() -> Self {
        Self {
            epoch: 1,
            initial_epoch: 1,
            train_data: Default::default(),
            validation_data: Default::default(),
            validation_rate: 0.0,
            batch_size: 32,
            shuffle: true,
            parallel: false,
            update_strategy: UpdateStrategy::MiniBatch(1),
            update_async: false,
        }
    }
}

impl<T> TrainConfig<T> {
    pub fn build(self) -> Train<T> {
        Train::new(self)
    }
}

pub struct Train<T> {
    pub config: TrainConfig<T>,
    rng: rand_isaac::Isaac64Rng,
    shuffle_table: Vec<usize>,
    pub epoch: usize,
}

impl<T> Train<T> {
    pub fn new(config: TrainConfig<T>) -> Self {
        assert!(config.epoch > 0);
        assert!(config.initial_epoch > 0);
        assert!(0.0 <= config.validation_rate && config.validation_rate <= 1.0);
        assert!(config.batch_size > 0);

        Self {
            rng: rand_isaac::Isaac64Rng::seed_from_u64(42),
            shuffle_table: (0..config.train_data.len()).collect(),
            epoch: config.initial_epoch - 1,
            config,
        }
    }

    pub fn is_end(&self) -> bool {
        self.epoch >= self.config.epoch
    }
}

impl<T: Sync + Send> Train<T> {
    pub fn fit_one_epoch<F>(&mut self, f: F)
    where
        F: Fn(&[&T], &mut TrainContext) + Sync + Send,
    {
        self.epoch += 1;
        if self.config.shuffle {
            self.shuffle_table.shuffle(&mut self.rng);
        }

        let do_validation = self.epoch
            == ((self.epoch as f32 * self.config.validation_rate).ceil()
                / self.config.validation_rate)
                .floor() as usize;

        if self.config.parallel {
            // train
            let mut ctx = self.context(true);
            let samples_threshold = match self.config.update_strategy {
                UpdateStrategy::Sample(n) => n,
                UpdateStrategy::MiniBatch(n) => n * self.config.batch_size,
                UpdateStrategy::Batch => usize::MAX,
            };
            let progress = Arc::new(Mutex::new(Progress::new(self.config.train_data.len())));

            let (metrics, mut ga, _) = self
                .shuffle_table
                .par_chunks(self.config.batch_size)
                .map(|shuffle_table| {
                    let data = shuffle_table
                        .iter()
                        .map(|i| &self.config.train_data[*i])
                        .collect::<Vec<_>>();
                    let mut ctx = ctx.child();
                    f(&data, &mut ctx);
                    let samples = ctx.metrics.total;
                    progress.lock().unwrap().update(samples);
                    (ctx.metrics, ctx.gradients_accumulator, samples)
                })
                .reduce(
                    || (Metrics::new(), GradientsAccumulator::new(), 0),
                    |mut a, b| {
                        a.0.merge(b.0);
                        a.1.merge(b.1);
                        a.2 += b.2;
                        if a.2 > samples_threshold {
                            a.1.optimize();
                            a.2 = 0;
                        }
                        a
                    },
                );
            ctx.merge_metrics(metrics);
            ga.optimize();
            ctx.print_result();

            // validation
            if !do_validation {
                return;
            }

            let mut ctx = self.context(false);
            let progress = Arc::new(Mutex::new(Progress::new(self.config.validation_data.len())));
            let metrics = self
                .config
                .validation_data
                .par_chunks(self.config.batch_size)
                .map(|data| {
                    let data: Vec<_> = data.iter().collect();
                    let mut ctx = ctx.child();
                    f(&data, &mut ctx);
                    progress.lock().unwrap().update(ctx.metrics.total);
                    ctx.metrics
                })
                .reduce(
                    || Metrics::new(),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                );
            ctx.merge_metrics(metrics);
            ctx.print_result();
        } else {
            // train
            let mut ctx = self.context(true);
            for shuffle_table in self.shuffle_table.chunks(self.config.batch_size) {
                let data = shuffle_table
                    .iter()
                    .map(|i| &self.config.train_data[*i])
                    .collect::<Vec<_>>();
                f(&data, &mut ctx);
                ctx.gradients_accumulator.optimize();
                ctx.print_progress();
            }
            ctx.print_result();

            // validation
            if !do_validation {
                return;
            }

            let mut ctx = self.context(false);
            for data in self.config.validation_data.chunks(self.config.batch_size) {
                let data: Vec<_> = data.iter().collect();
                f(&data, &mut ctx);
                ctx.print_progress();
            }
            ctx.print_result();
        }
    }

    pub fn fit<F>(&mut self, f: F)
    where
        F: Fn(&[&T], &mut TrainContext) + Sync + Send,
    {
        while !self.is_end() {
            self.fit_one_epoch(&f);
        }
    }

    fn context(&self, train: bool) -> TrainContext {
        TrainContext {
            total: Some(if train {
                self.config.train_data.len()
            } else {
                self.config.validation_data.len()
            }),
            epoch: self.epoch,
            train,
            metrics: Metrics::new(),
            time: std::time::Instant::now(),
            gradients_accumulator: GradientsAccumulator::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    /// Update after processing the specified number of samples.
    Sample(usize),
    /// Update after processing the specified number of mini-batch.
    MiniBatch(usize),
    /// Update each epoch.
    Batch,
}

pub struct TrainContext {
    pub total: Option<usize>,
    pub epoch: usize,
    pub train: bool,
    pub metrics: Metrics,
    pub time: std::time::Instant,
    gradients_accumulator: GradientsAccumulator,
}

impl TrainContext {
    fn child(&self) -> Self {
        TrainContext {
            total: self.total,
            epoch: self.epoch,
            train: self.train,
            metrics: Metrics::new(),
            time: self.time,
            gradients_accumulator: GradientsAccumulator::new(),
        }
    }

    pub fn finish_batch(&mut self, loss: &Tensor, n: usize) {
        self.optimize(loss);
        self.count(n);
        self.add_metric(metrics::Loss::new(loss[[]], n));
    }

    pub fn optimize(&mut self, loss: &Tensor) {
        if self.train {
            self.gradients_accumulator.compute(loss);
        }
    }

    pub fn count(&mut self, n: usize) {
        self.metrics.count(n);
    }

    pub fn add_metric<T: Metric>(&mut self, metric: T) {
        self.metrics.add(metric);
    }

    pub fn merge_metrics(&mut self, metrics: Metrics) {
        self.metrics.merge(metrics);
    }

    pub fn print_progress(&self) {
        print!("\x1b[2K");
        if let Some(total) = self.total {
            print!("{:>6.2}%", self.metrics.total as f32 * 100.0 / total as f32);
        } else {
            print!("{}", self.metrics.total);
        }
        print!("\r");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    pub fn print_result(&self) {
        println!(
            "{} epoch: {}, {}time: {:.2}s",
            if self.train { "train" } else { "valid" },
            self.epoch,
            self.metrics.display_metrics(),
            self.time.elapsed().as_secs_f32()
        );
    }
}

struct Progress {
    total: usize,
    processed: usize,
    last_show: std::time::Instant,
}

impl Progress {
    fn new(total: usize) -> Self {
        Self {
            total,
            processed: 0,
            last_show: std::time::Instant::now(),
        }
    }

    fn update(&mut self, n: usize) {
        self.processed += n;
        if self.last_show.elapsed().as_millis() >= 100 {
            self.last_show = std::time::Instant::now();
            print!("\x1b[2K");
            print!(
                "{:>6.2}%",
                self.processed as f32 * 100.0 / self.total as f32
            );
            print!("\r");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
}

#[test]
fn test() {
    TrainConfig {
        epoch: 10,
        train_data: (0..100).collect(),
        validation_data: (0..10).collect(),
        validation_rate: 0.25,
        batch_size: 10,
        parallel: false,
        update_async: false,
        ..Default::default()
    }
    .build()
    .fit(|batch, ctx| {
        std::thread::sleep(std::time::Duration::from_millis(10));
        let loss = Tensor::new(scalar(0.0));
        ctx.finish_batch(&loss, batch.len());
        ctx.add_metric(metrics::Loss::new(loss[[]], batch.len()));
    })
}
