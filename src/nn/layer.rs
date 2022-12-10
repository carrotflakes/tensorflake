use crate::*;

pub trait Layer: 'static {
    type Input;
    type Output;

    fn call(&self, input: Self::Input, train: bool) -> Self::Output;
    fn all_params(&self) -> Vec<ParamNDA>;

    fn name(&self) -> &'static str {
        let name = std::any::type_name::<Self>();
        name.split("::").last().unwrap_or(name)
    }

    fn then<T: Layer<Input = Self::Output>>(self, next: T) -> Then<Self, T>
    where
        Self: Sized,
    {
        Then::new(self, next)
    }

    fn then_fn<O>(
        self,
        f: &'static (dyn Fn(&Self::Output) -> O + Sync + Send),
    ) -> Then<Self, FnAsLayer<Self::Output, O>>
    where
        Self: Sized,
    {
        Then::new(self, FnAsLayer(f))
    }
}

pub struct Then<T: Layer, U: Layer<Input = T::Output>> {
    pub first: T,
    pub second: U,
}

impl<T: Layer, U: Layer<Input = T::Output>> Then<T, U> {
    pub fn new(first: T, second: U) -> Self {
        Self { first, second }
    }
}

impl<T: Layer, U: Layer<Input = T::Output>> Layer for Then<T, U> {
    type Input = T::Input;
    type Output = U::Output;

    fn call(&self, x: Self::Input, train: bool) -> Self::Output {
        self.second.call(self.first.call(x, train), train)
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        self.first
            .all_params()
            .into_iter()
            .chain(self.second.all_params())
            .collect()
    }
}

pub struct FnAsLayer<I: 'static, O: 'static>(pub &'static (dyn Fn(&I) -> O + Sync + Send));

impl<I: 'static, O: 'static> Layer for FnAsLayer<I, O> {
    type Input = I;
    type Output = O;

    fn call(&self, x: Self::Input, _train: bool) -> Self::Output {
        (self.0)(&x)
    }

    fn all_params(&self) -> Vec<ParamNDA> {
        vec![]
    }
}

#[test]
fn test() {
    let init = initializers::InitializerWithOptimizer::new(
        ndarray_rand::rand_distr::Normal::new(0.0, 0.1).unwrap(),
        optimizers::Adam::new(),
    );
    let l1 = nn::Linear::new(2, 3, init.clone(), Some(init.clone()));
    let l2 = nn::Linear::new(3, 2, init.clone(), Some(init.clone()));
    let fnn: &dyn Layer<Input = ComputedNDA, Output = ComputedNDA> =
        &l1.then_fn(&nn::activations::relu).then(l2);
    let y = fnn.call(backprop(NDArray::ones(&[1, 2][..])), false);
    assert_eq!(y.shape(), &[1, 2]);
}
