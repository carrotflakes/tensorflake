use crate::*;

pub trait Layer: Sync + Send + 'static {
    type Input;
    type Output;

    fn call(&self, input: Self::Input, train: bool) -> Self::Output;
    fn all_params(&self) -> Vec<Param>;

    fn name(&self) -> &'static str {
        let name = std::any::type_name::<Self>();
        name.split("::").last().unwrap_or(name)
    }
}
