use crate::{
    initializers::{Initializer, Scope},
    *,
};

use super::{activations::softmax, normalization::Normalization, Linear};

pub struct MultiHeadAttention {
    head_dim: usize,
    num_heads: usize,

    key_proj: Linear,
    query_proj: Linear,
    value_proj: Linear,
    // TODO: dropout
}

impl MultiHeadAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        w: impl Initializer<ParamNDA> + Scope,
        b: impl Initializer<ParamNDA> + Scope,
    ) -> Self {
        assert!(embed_dim % num_heads == 0);

        MultiHeadAttention {
            head_dim: embed_dim / num_heads,
            num_heads,
            key_proj: Linear::new(
                embed_dim,
                embed_dim,
                w.scope("key_proj"),
                Some(b.scope("key_proj")),
            ),
            query_proj: Linear::new(
                embed_dim,
                embed_dim,
                w.scope("query_proj"),
                Some(b.scope("query_proj")),
            ),
            value_proj: Linear::new(
                embed_dim,
                embed_dim,
                w.scope("value_proj"),
                Some(b.scope("value_proj")),
            ),
        }
    }

    pub fn call(&self, x: &ComputedNDA, attn_mask: &ComputedNDA, train: bool) -> ComputedNDA {
        // (N, L, E) -> (N, L, num_heads * head_dim)
        let query = self.query_proj.call(x.clone(), train);
        let key = self.key_proj.call(x.clone(), train);
        let value = self.value_proj.call(x.clone(), train);

        // (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        let query = self.separate_heads(query);
        let key = self.separate_heads(key);
        let value = self.separate_heads(value);

        // Calculate the attention scores
        // (N, num_heads, L, head_dim) * (N, num_head, head_dim, L) -> (N, num_head, L, L)
        let attention =
            query.matmul(&key.mat_t()) / ComputedNDA::new(scalar((self.head_dim as f32).sqrt()));

        // Apply softmax to the attention scores
        let attention = softmax(&(attention + self.extend_mask(attn_mask)));

        // Applying attention weights
        // (N, num_heads, L, L) * (N, num_heads, L, head_dim) -> (N, num_heads, L, head_dim)
        let attention_value = attention.matmul(&value);

        // (N, num_heads, L, head_dim) -> (N, L, num_heads * head_dim)
        let y = self.merge_heads(attention_value);

        y
    }

    fn separate_heads(&self, features: ComputedNDA) -> ComputedNDA {
        // (N, L, num_heads * head_dim) -> (N, L, num_heads, head_dim)
        let batch_size = features.shape()[0];
        let input_len = features.shape()[1];

        let features = features.reshape([batch_size, input_len, self.num_heads, self.head_dim]);

        // (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        features.transpose(vec![0, 2, 1, 3])
    }

    fn merge_heads(&self, features: ComputedNDA) -> ComputedNDA {
        // (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        let features = features.transpose(vec![0, 2, 1, 3]);

        // (N, L, num_heads, head_dim) -> (N, L, num_heads * head_dim)
        let batch_size = features.shape()[0];
        let input_len = features.shape()[1];

        // features.reshape([batch_size, input_len, 0])
        features.reshape([batch_size, input_len, self.num_heads * self.head_dim])
    }

    fn extend_mask(&self, mask: &ComputedNDA) -> ComputedNDA {
        // (N, L) -> (N, 1, 1, L)

        let batch_size = mask.shape()[0];
        let input_len = mask.shape()[1];

        let extended_mask = mask.reshape([batch_size, 1, 1, input_len]);

        // Adding -1e5 makes masked locations zeroed out during softmax
        (ComputedNDA::new(scalar(1.0)) - extended_mask) * ComputedNDA::new(scalar(-1e5))
    }
}

pub struct MHAAddNorm {
    attention: MultiHeadAttention,
    dense: Linear,
    norm: Normalization,
}

impl MHAAddNorm {
    pub fn new(
        dim: usize,
        num_heads: usize,
        layer_norm_eps: f32,
        w: impl Initializer<ParamNDA> + Scope,
        b: impl Initializer<ParamNDA> + Scope,
        opt: impl Optimizer<NDArray> + Clone,
    ) -> Self {
        Self {
            attention: MultiHeadAttention::new(dim, num_heads, w.scope("mha"), b.scope("mha")),
            dense: Linear::new(dim, dim, w.scope("dense"), Some(b.scope("dense"))),
            norm: Normalization::new(vec![1], vec![dim], layer_norm_eps, opt),
        }
    }

    pub fn call(&self, x: &ComputedNDA, attn_mask: &ComputedNDA, train: bool) -> ComputedNDA {
        let y = self.attention.call(x, attn_mask, train);
        let y = self.dense.call(y, train);
        self.norm.call(&y + x, train)
    }
}

#[test]
fn test() {
    use ndarray_rand::rand_distr::Uniform;

    let init = initializers::with_optimizer::InitializerWithOptimizer::new(
        initializers::random_initializer::RandomInitializer::new(Uniform::new(-0.01, 0.01)),
        optimizers::Adam::new(),
    );

    let mha = MHAAddNorm::new(
        64,
        4,
        1e-5,
        init.scope("mha"),
        init.scope("mha"),
        optimizers::Adam::new(),
    );

    let x = ComputedNDA::new(
        NDArray::from_shape_vec(
            &[3, 8, 64][..],
            (0..64 * 8 * 3)
                .map(|x| (x % 109) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let attn_mask = ComputedNDA::new(
        NDArray::from_shape_vec(&[3, 8][..], (0..8 * 3).map(|_| 1.0).collect::<Vec<_>>()).unwrap(),
    );

    let y = mha.call(&x, &attn_mask, true);
    assert_eq!(y.shape(), x.shape());
    // dbg!(&*y);
}
