use crate::*;

pub trait Layer {
    fn call(&self, xs: Vec<Tensor>, train: bool) -> Vec<Tensor>
    where
        Self: Sized + 'static;

    fn all_params(&self) -> Vec<Param>;
}
