# TensorFlake

TBD

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
- [ ] Tensordot
- [ ] Transposed convolution
- [ ] Batch normalization
- [ ] Benchmarks
- [ ] Measure the execution time of functions and export it as dot file
- [ ] Examples
  - [ ] CNN -> too slow...
  - [ ] RNN
  - [ ] GAN
  - [ ] VAE
- [ ] wasm
- [x] safe rust

## Author

* carrotflakes (carrotflakes@gmail.com)

## Copyright

Copyright (c) 2022 carrotflakes (carrotflakes@gmail.com)

## License

Licensed under the MIT License.
