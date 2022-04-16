# TensorFlake

TensorFlake is my own machine learning framework.
This aims to be able to train several networks on the CPU fast.

## TODO

- [ ] smallvec
- [x] backward without create_graph
- [x] Efficient Linear -> matmul_add
- [ ] Save & load -> param_bin.rs
  - [ ] Save the structure -> serde?
- [x] Strong typing -> Functional API (tensorflake::function::chain)
  - [ ] Generic for dimension
- [ ] Multi thread
  - [x] High level API
  - [ ] Synchronous update
  - [x] Asynchronous update
- [x] Impl ops
- [ ] Lazy execution for optimization on the graph
- [x] Regularization
- [x] Tensordot -> [ndarray_einsum_beta](https://crates.io/crates/ndarray_einsum_beta)
- [x] Transposed convolution
- [ ] Batch normalization
- [x] Embedding
- [ ] Sequential
- [ ] Param creator -> Initializer
- [ ] Benchmarks
- [ ] Measure the execution time of functions and export it as dot file
- [ ] Tensor summarization
- [ ] Sparse tensor
- [ ] Examples
  - [ ] CNN
  - [ ] RNN
  - [ ] GAN
  - [ ] VAE
- [ ] wasm
- [x] safe rust
- [ ] Compare performance against GPU Tensorflow
- [ ] Fitting report
- [ ] Generic ndarray
- [ ] functions
  - [ ] reduce_sum
  - [ ] signum
- [ ] Optimize consecutive element-wise operations

## Benchmark

``` sh
$ cargo +nightly bench -q > benches/result.txt
```

## Author

* carrotflakes (carrotflakes@gmail.com)

## Copyright

Copyright (c) 2022 carrotflakes (carrotflakes@gmail.com)

## License

Licensed under the MIT License.
