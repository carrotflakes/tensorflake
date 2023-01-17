pub mod random_initializer;
pub mod with_optimizer;
pub mod with_shared_optimizer;

pub trait Initializer<T: Send + Sync> {
    fn initialize(&self, shape: &[usize]) -> T;
}

pub trait Scope {
    fn scope(&self, name: impl ToString) -> Self;
}
