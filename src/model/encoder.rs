/// Transformer encoder stack built from scratch using Burn primitives.
///
/// Architecture per layer (pre-norm):
///   x = x + SelfAttention(LayerNorm(x))
///   x = x + FeedForward(LayerNorm(x))
///
/// The full encoder stacks `num_layers` such blocks (minimum 6).
use burn::{
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, activation::gelu, Bool, Float, Tensor},
    config::Config,
};

// ── Per-layer config ──────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TransformerEncoderLayerConfig {
    pub d_model:   usize,
    pub num_heads: usize,
    pub d_ff:      usize,
    #[config(default = "0.1")]
    pub dropout:   f64,
}

impl TransformerEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoderLayer<B> {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttentionConfig::new(self.d_model, self.num_heads)
                .with_dropout(self.dropout)
                .init(device),
            ff1:    LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2:    LinearConfig::new(self.d_ff,    self.d_model).init(device),
            norm1:  LayerNormConfig::new(self.d_model).init(device),
            norm2:  LayerNormConfig::new(self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

// ── Single encoder layer ──────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    /// Multi-head self-attention.
    self_attn: MultiHeadAttention<B>,
    /// Feed-forward: linear 1 (expand).
    ff1:       Linear<B>,
    /// Feed-forward: linear 2 (project back).
    ff2:       Linear<B>,
    /// Layer norms (pre-norm style).
    norm1:     LayerNorm<B>,
    norm2:     LayerNorm<B>,
    dropout:   Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    /// `x`    : [batch, seq_len, d_model]
    /// `mask` : [batch, seq_len]  — 1 for real tokens, 0 for padding (Int)
    ///
    /// Returns : [batch, seq_len, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3, Float> {
        // ── Self-attention sub-layer (pre-norm) ───────────────────────────────
        let residual = x.clone();
        let x_norm   = self.norm1.forward(x);

        let mha_out = self.self_attn.forward(
            MhaInput::new(x_norm.clone(), x_norm.clone(), x_norm)
                .mask_pad(pad_mask.clone()),
        );
        let x = residual + self.dropout.forward(mha_out.context);

        // ── Feed-forward sub-layer (pre-norm) ─────────────────────────────────
        let residual = x.clone();
        let x_norm   = self.norm2.forward(x);

        let x_ff = self.ff1.forward(x_norm);
        let x_ff = gelu(x_ff);
        let x_ff = self.dropout.forward(x_ff);
        let x_ff = self.ff2.forward(x_ff);

        residual + self.dropout.forward(x_ff)
    }
}

// ── Full encoder stack ────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TransformerEncoderConfig {
    pub d_model:    usize,
    pub num_heads:  usize,
    pub d_ff:       usize,
    /// Minimum 6 as required by the assignment.
    #[config(default = "6")]
    pub num_layers: usize,
    #[config(default = "0.1")]
    pub dropout:    f64,
}

impl TransformerEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoder<B> {
        let layer_cfg = TransformerEncoderLayerConfig::new(
            self.d_model, self.num_heads, self.d_ff,
        )
        .with_dropout(self.dropout);

        let layers: Vec<TransformerEncoderLayer<B>> = (0..self.num_layers)
            .map(|_| layer_cfg.init(device))
            .collect();

        TransformerEncoder { layers }
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    /// Stack of encoder layers — at least 6 (per assignment requirements).
    layers: Vec<TransformerEncoderLayer<B>>,
}

impl<B: Backend> TransformerEncoder<B> {
    /// `x`    : [batch, seq_len, d_model]
    /// `mask` : [batch, seq_len]  padding mask (Int 0/1)
    ///
    /// Returns : [batch, seq_len, d_model]
    pub fn forward(
        &self,
        mut x: Tensor<B, 3, Float>,
        mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3, Float> {
        for layer in &self.layers {
            x = layer.forward(x, mask.clone());
        }
        x
    }
}
