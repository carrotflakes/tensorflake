# TensorFlake

TBD

## TODO

- [ ] smallvec
- [x] backward without create_graph
- [x] Is Backward removable? -> currently NO because the Function trait is not object-safe.
- [x] Adam
- [ ] Efficient Linear
- [ ] Save & load -> param_bin.rs
- [x] Dropout
- [x] Strong typing -> Functional API (tensorflake::function::chain)
- [ ] Multi thread -> see examples/coin.rs
  - [ ] High level API
- [x] Impl ops
- [ ] Lazy execution
- [ ] Regularization
  - [ ] Integrate to optimizee
- [ ] Batch normalization
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
