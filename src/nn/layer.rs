use crate::*;

pub trait Layer {
    fn call(&self, xs: Vec<Variable>, train: bool) -> Vec<Variable>
    where
        Self: Sized + 'static;

    fn all_params(&self) -> Vec<Variable>;
}
