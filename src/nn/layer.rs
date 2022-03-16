use crate::*;

pub trait Layer {
    fn call<const ENABLE_BACKPROP: bool>(
        &self,
        xs: Vec<Variable<ENABLE_BACKPROP>>,
    ) -> Vec<Variable<ENABLE_BACKPROP>>
    where
        Self: Sized + 'static;

    fn all_params(&self) -> Vec<Variable<ENABLE_BACKPROP>>;

    fn clear_grads(&self) {
        for param in self.all_params() {
            param.clear_grad();
        }
    }
}
