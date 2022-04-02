# TensorFlake

TensorFlake is my own machine learning framework.
I'm aiming to be able to train several networks on the CPU fast.

## TODO

- [ ] smallvec
- [x] backward without create_graph
- [x] Is Backward removable? -> currently NO because the Function trait is not object-safe.
- [x] Adam
- [x] Efficient Linear -> matmul_add
- [ ] Save & load -> param_bin.rs
  - [ ] Save the structure
- [x] Dropout
- [x] Strong typing -> Functional API (tensorflake::function::chain)
- [ ] Multi thread -> see examples/coin.rs
  - [ ] High level API
  - [ ] Synchronous update
  - [x] Asynchronous update
- [x] Impl ops
- [ ] Lazy execution for optimization on the graph
- [x] Regularization
- [x] Tensordot -> [ndarray_einsum_beta](https://crates.io/crates/ndarray_einsum_beta)
- [ ] Transposed convolution
- [ ] Batch normalization
- [ ] Embedding
- [ ] Benchmarks
- [ ] Measure the execution time of functions and export it as dot file
- [ ] Tensor summarization
- [ ] Examples
  - [ ] CNN -> too slow...
  - [ ] RNN
  - [ ] GAN
  - [ ] VAE
- [ ] wasm
- [x] safe rust
- [ ] Compare performance against GPU Tensorflow
- [ ] Fitting report

## Author

* carrotflakes (carrotflakes@gmail.com)

## Copyright

Copyright (c) 2022 carrotflakes (carrotflakes@gmail.com)

## License

Licensed under the MIT License.
