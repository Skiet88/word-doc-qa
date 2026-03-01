/// Token embeddings + sinusoidal positional embeddings.
///
/// Input  : [batch, seq_len]  integer token IDs
/// Output : [batch, seq_len, d_model]  float embeddings
use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Float, Int, Tensor},
    config::Config,
};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct EmbeddingsConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

impl EmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Embeddings<B> {
        let token_emb = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_emb   = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);
        let norm      = LayerNormConfig::new(self.d_model).init(device);
        let dropout   = DropoutConfig::new(self.dropout).init();

        Embeddings { token_emb, pos_emb, norm, dropout }
    }
}

// ── Module ────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Embeddings<B: Backend> {
    token_emb: Embedding<B>,
    pos_emb:   Embedding<B>,
    norm:      LayerNorm<B>,
    dropout:   Dropout,
}

impl<B: Backend> Embeddings<B> {
    /// `token_ids` : [batch, seq_len]
    /// Returns      : [batch, seq_len, d_model]
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3, Float> {
        let [batch, seq_len] = token_ids.dims();

        // Build position indices [0, 1, 2, … seq_len-1] and broadcast to [batch, seq_len].
        let pos: Vec<i32> = (0..seq_len as i32).collect();
        let pos_ids = Tensor::<B, 1, Int>::from_ints(pos.as_slice(), &token_ids.device())
            .unsqueeze::<2>()                        // [1, seq_len]
            .expand([batch as i64, seq_len as i64]); // [batch, seq_len]

        let tok = self.token_emb.forward(token_ids); // [batch, seq, d]
        let pos = self.pos_emb.forward(pos_ids);     // [batch, seq, d]

        let x = tok + pos;
        let x = self.norm.forward(x);
        self.dropout.forward(x)
    }
}
