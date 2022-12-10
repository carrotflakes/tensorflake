use crate::{regularizers::Regularizer, *};

#[derive(Clone)]
pub struct WithRegularization<O: Optimizer<NDArray>, R: Regularizer> {
    pub optimizer: O,
    pub regularizer: R,
}

impl<O: Optimizer<NDArray>, R: Regularizer> WithRegularization<O, R> {
    pub fn new(optimizer: O, regularizer: R) -> Self {
        Self {
            optimizer,
            regularizer,
        }
    }
}

impl<O: Optimizer<NDArray>, R: Regularizer> Optimizer<NDArray> for WithRegularization<O, R> {
    type State = O::State;

    fn new_state(&self, shape: &[usize]) -> Self::State {
        self.optimizer.new_state(shape)
    }

    fn update(&mut self, data: &mut NDArray, state: &mut Self::State, grad: &NDArray) {
        let grad = (grad + &*self.regularizer.grad(&data.clone().into())).into_ndarray();
        self.optimizer.update(data, state, &grad);
    }
}

#[test]
fn test() {
    let optimizer = super::Adam::new_with_params(0.01, 0.9, 0.999);
    let optimizer = WithRegularization::new(optimizer, regularizers::L1::new(0.01));
    super::test_optimizer(optimizer);
}
