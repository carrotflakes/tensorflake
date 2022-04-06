use ndarray_rand::rand::prelude::*;
use rayon::prelude::*;
use tensorflake::*;

#[derive(Clone)]
pub struct TrainingConfig<T> {
    pub epoch: usize,
    pub initial_epoch: usize,
    pub train_data: Vec<T>,
    pub validation_data: Vec<T>,
    pub validation_rate: f32,
    pub batch_size: usize,
    pub shuffle: bool,
    pub parallel: bool,
    pub update_async: bool,
}

impl<T> Default for TrainingConfig<T> {
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
            update_async: false,
        }
    }
}

impl<T> TrainingConfig<T> {
    pub fn build(self) -> Training<T> {
        Training::new(self)
    }
}

pub struct Training<T> {
    pub config: TrainingConfig<T>,
    rng: rand_isaac::Isaac64Rng,
    shuffle_table: Vec<usize>,
    pub epoch: usize,
}

impl<T> Training<T> {
    pub fn new(config: TrainingConfig<T>) -> Self {
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

impl<T: Sync + Send> Training<T> {
    pub fn fit_one_epoch<F>(&mut self, f: F)
    where
        F: Fn(&[&T], &mut ExecutionContext) + Sync + Send,
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
            let mut ctx = ExecutionContext {
                total: Some(self.config.train_data.len()),
                epoch: self.epoch,
                train: true,
                metrics: Metrics::new(),
                time: std::time::Instant::now(),
            };
            let metrics = self
                .shuffle_table
                .par_chunks(self.config.batch_size)
                .map(|shuffle_table| {
                    let data = shuffle_table
                        .iter()
                        .map(|i| &self.config.train_data[*i])
                        .collect::<Vec<_>>();
                    let mut ctx = ExecutionContext {
                        total: ctx.total,
                        epoch: ctx.epoch,
                        train: ctx.train,
                        metrics: Metrics::new(),
                        time: ctx.time,
                    };
                    f(&data, &mut ctx);
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

            // validation
            if !do_validation {
                return;
            }

            let mut ctx = ExecutionContext {
                total: Some(self.config.train_data.len()),
                epoch: self.epoch,
                train: false,
                metrics: Metrics::new(),
                time: std::time::Instant::now(),
            };
            let metrics = self
                .config
                .validation_data
                .par_chunks(self.config.batch_size)
                .map(|data| {
                    let data: Vec<_> = data.iter().collect();
                    let mut ctx = ExecutionContext {
                        total: ctx.total,
                        epoch: ctx.epoch,
                        train: ctx.train,
                        metrics: Metrics::new(),
                        time: ctx.time,
                    };
                    f(&data, &mut ctx);
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
            let mut ctx = ExecutionContext {
                total: Some(self.config.train_data.len()),
                epoch: self.epoch,
                train: true,
                metrics: Metrics::new(),
                time: std::time::Instant::now(),
            };
            for shuffle_table in self.shuffle_table.chunks(self.config.batch_size) {
                let data = shuffle_table
                    .iter()
                    .map(|i| &self.config.train_data[*i])
                    .collect::<Vec<_>>();
                f(&data, &mut ctx);
                ctx.print_progress();
            }
            ctx.print_result();

            // validation
            if !do_validation {
                return;
            }

            let mut ctx = ExecutionContext {
                total: Some(self.config.validation_data.len()),
                epoch: self.epoch,
                train: false,
                metrics: Metrics::new(),
                time: std::time::Instant::now(),
            };
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
        F: Fn(&[&T], &mut ExecutionContext) + Sync + Send,
    {
        while !self.is_end() {
            self.fit_one_epoch(&f);
        }
    }
}

#[allow(dead_code)]
fn main() {
    TrainingConfig {
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
        std::thread::sleep(std::time::Duration::from_millis(50));
        ctx.count(batch.len());
        ctx.add_metric(metrics::Loss::new(0.0, batch.len()));
    })
}
