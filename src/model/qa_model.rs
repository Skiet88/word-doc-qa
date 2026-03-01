/// Full extractive QA model.
///
/// Architecture:
///   Input tokens → Embeddings → 6-layer TransformerEncoder
///                              → Start-logit head (Linear)
///                              → End-logit   head (Linear)
///
/// Training objective: cross-entropy on start position + cross-entropy on end position.
use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

use crate::data::dataset::{QaBatch, MAX_SEQ_LEN};
use crate::model::{
    embeddings::{Embeddings, EmbeddingsConfig},
    encoder::{TransformerEncoder, TransformerEncoderConfig},
};

// ── Hyperparameter config ─────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct QaModelConfig {
    pub vocab_size:  usize,
    /// Hidden dimension (d_model).
    #[config(default = "256")]
    pub d_model:     usize,
    /// Number of attention heads — must divide d_model evenly.
    #[config(default = "8")]
    pub num_heads:   usize,
    /// Feed-forward intermediate dimension.
    #[config(default = "1024")]
    pub d_ff:        usize,
    /// Number of encoder layers — kept at 6 (assignment minimum).
    #[config(default = "6")]
    pub num_layers:  usize,
    #[config(default = "0.1")]
    pub dropout:     f64,
}

impl QaModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QaModel<B> {
        let embeddings = EmbeddingsConfig::new(self.vocab_size, self.d_model, MAX_SEQ_LEN)
            .with_dropout(self.dropout)
            .init(device);

        let encoder = TransformerEncoderConfig::new(self.d_model, self.num_heads, self.d_ff)
            .with_num_layers(self.num_layers)
            .with_dropout(self.dropout)
            .init(device);

        // Each head maps d_model → 1 logit per token position.
        let start_head = LinearConfig::new(self.d_model, 1).init(device);
        let end_head   = LinearConfig::new(self.d_model, 1).init(device);

        QaModel { embeddings, encoder, start_head, end_head }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct QaModel<B: Backend> {
    embeddings: Embeddings<B>,
    encoder:    TransformerEncoder<B>,
    /// Projects each token's hidden state to a start-position logit.
    start_head: Linear<B>,
    /// Projects each token's hidden state to an end-position logit.
    end_head:   Linear<B>,
}

/// Output of a forward pass.
#[derive(Debug)]
pub struct QaOutput<B: Backend> {
    /// Scalar training loss (sum of start and end cross-entropies / 2).
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> QaModel<B> {
    /// Run the model on a batch and compute loss.
    ///
    /// `input_ids`       : [batch, seq_len]  Int
    /// `attn_mask`       : [batch, seq_len]  Int (1 = real, 0 = pad)
    /// `start_positions` : [batch]           Int
    /// `end_positions`   : [batch]           Int
    pub fn forward(&self, batch: QaBatch<B>) -> QaOutput<B> {
        let device = batch.input_ids.device();
        let [_batch_size, _seq_len] = batch.input_ids.dims();

        // ── Embeddings ────────────────────────────────────────────────────────
        let x = self.embeddings.forward(batch.input_ids); // [B, S, D]

        // ── Build boolean padding mask (true = padding position to ignore) ────
        let pad_mask = batch.attn_mask
            .equal_elem(0_i64)   // [B, S]  Bool — true where padding
            ;

        // ── Transformer encoder ───────────────────────────────────────────────
        let hidden = self.encoder.forward(x, pad_mask); // [B, S, D]

        // ── Output projection ─────────────────────────────────────────────────
        // start_head / end_head: [B, S, D] → [B, S, 1] → flatten last dim → [B, S]
        let start_logits = self.start_head.forward(hidden.clone())
            .flatten::<2>(1, 2);  // [B, S]
        let end_logits = self.end_head.forward(hidden)
            .flatten::<2>(1, 2);  // [B, S]

        // ── Loss ──────────────────────────────────────────────────────────────
        // CrossEntropyLoss expects logits [B, C] and targets [B].
        // Here C = seq_len, so no reshaping needed.
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let start_loss = loss_fn.forward(start_logits.clone(), batch.start_positions);
        let end_loss   = loss_fn.forward(end_logits.clone(),   batch.end_positions);
        let loss = (start_loss + end_loss) / 2.0_f64;

        let _ = (start_logits, end_logits); // used only to compute loss above
        QaOutput { loss }
    }

    // ── Inference-only forward (no loss) ──────────────────────────────────────

    /// Returns `(start_idx, end_idx)` predicted token positions for each item
    /// in the batch.
    pub fn predict(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attn_mask: Tensor<B, 2, Int>,
    ) -> Vec<(usize, usize)> {
        let [batch_size, _seq_len] = input_ids.dims();
        let x    = self.embeddings.forward(input_ids.clone());
        let mask = attn_mask.equal_elem(0_i64);
        let h    = self.encoder.forward(x, mask);

        let start_logits = self.start_head.forward(h.clone()).flatten::<2>(1, 2); // [B, S]
        let end_logits   = self.end_head.forward(h).flatten::<2>(1, 2);           // [B, S]

        // Argmax along the sequence dimension.
        let starts = start_logits.argmax(1); // [B, 1]
        let ends   = end_logits.argmax(1);   // [B, 1]

        // Move to CPU to read scalars.
        let s_data = starts.into_data();
        let e_data = ends.into_data();

        (0..batch_size)
            .map(|i| {
                let s = s_data.as_slice::<i64>().map(|sl| sl[i]).unwrap_or(0) as usize;
                let e = e_data.as_slice::<i64>().map(|sl| sl[i]).unwrap_or(0) as usize;
                let e = e.max(s); // end must be ≥ start
                (s, e)
            })
            .collect()
    }
}
