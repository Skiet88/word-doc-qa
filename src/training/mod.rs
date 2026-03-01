/// Training pipeline.
///
/// Uses Burn's manual training loop with the Autodiff backend so that
/// gradient flow, checkpointing, and metrics are all explicit and easy to follow.
use std::path::Path;
use std::time::Instant;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    data::dataset::Dataset,
    module::AutodiffModule,
    prelude::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
};
use serde::{Deserialize, Serialize};

use crate::data::{
    dataset::{build_datasets, QaBatcher},
    loader::load_all_docx,
    tokenizer::build_vocab,
};
use crate::model::qa_model::{QaModel, QaModelConfig};

// ── Training hyperparameters ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub vocab_size:    usize,
    pub d_model:       usize,
    pub num_heads:     usize,
    pub d_ff:          usize,
    pub num_layers:    usize,
    pub dropout:       f64,
    pub learning_rate: f64,
    pub batch_size:    usize,
    pub num_epochs:    usize,
    pub valid_ratio:   f64,
    pub seed:          u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size:    8_000,
            d_model:       256,
            num_heads:     8,
            d_ff:          1_024,
            num_layers:    6,
            dropout:       0.1,
            learning_rate: 1e-4,
            batch_size:    8,
            num_epochs:    50,
            valid_ratio:   0.1,
            seed:          42,
        }
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

type TrainBackend = Autodiff<Wgpu>;

/// Train the QA model on all .docx files found in `docs_dir`.
/// Checkpoints and the vocabulary are saved to `output_dir`.
pub fn train(docs_dir: &Path, output_dir: &Path, cfg: &TrainingConfig) {
    std::fs::create_dir_all(output_dir).expect("cannot create output directory");

    // ── 1. Load documents ─────────────────────────────────────────────────────
    println!("[train] Scanning '{}'…", docs_dir.display());
    let docs = load_all_docx(docs_dir);
    if docs.is_empty() {
        eprintln!("[train] No .docx files found. Aborting.");
        return;
    }
    let total_paragraphs: usize = docs.iter().map(|d| d.paragraphs.len()).sum();
    println!("[train] Loaded {} document(s), {} paragraph(s) total.", docs.len(), total_paragraphs);

    // ── 2. Build vocabulary ───────────────────────────────────────────────────
    let all_text: Vec<String> = docs.iter().flat_map(|d| d.paragraphs.clone()).collect();
    let vocab = build_vocab(&all_text, cfg.vocab_size);
    println!("[train] Vocabulary size: {}", vocab.len());
    vocab
        .save(&output_dir.join("vocab.json"))
        .expect("could not save vocab");

    // ── 3. Build datasets ─────────────────────────────────────────────────────
    let (train_ds, valid_ds) = build_datasets(&docs, &vocab, cfg.valid_ratio);
    println!(
        "[train] Dataset — train: {} items, validation: {} items.",
        train_ds.len(),
        valid_ds.len()
    );
    if train_ds.is_empty() {
        eprintln!("[train] No training items generated. Aborting.");
        return;
    }

    // ── 4. Initialise Burn backend and device ─────────────────────────────────
    let device = WgpuDevice::default();

    let batcher_train: QaBatcher<TrainBackend> = QaBatcher::new(device.clone());
    let batcher_valid: QaBatcher<Wgpu>         = QaBatcher::new(device.clone());

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(cfg.batch_size)
        .shuffle(cfg.seed)
        .num_workers(1)
        .build(train_ds);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(cfg.batch_size)
        .num_workers(1)
        .build(valid_ds);

    // ── 5. Initialise model and optimiser ─────────────────────────────────────
    // Save model config so it can be reloaded at inference time.
    let model_cfg = QaModelConfig::new(vocab.len())
        .with_d_model(cfg.d_model)
        .with_num_heads(cfg.num_heads)
        .with_d_ff(cfg.d_ff)
        .with_num_layers(cfg.num_layers)
        .with_dropout(cfg.dropout);

    let cfg_json = serde_json::to_string_pretty(&model_cfg)
        .expect("cannot serialise model config");
    std::fs::write(output_dir.join("model_config.json"), cfg_json)
        .expect("cannot write model config");

    let mut model: QaModel<TrainBackend> = model_cfg.init(&device);
    let mut optim = AdamConfig::new().init();

    // ── 6. Training loop ──────────────────────────────────────────────────────
    println!("[train] Starting training ({} epochs)…\n", cfg.num_epochs);

    for epoch in 1..=cfg.num_epochs {
        let t_start = Instant::now();
        let mut train_loss_sum = 0.0_f64;
        let mut train_batches  = 0_usize;

        // ── Training pass ─────────────────────────────────────────────────────
        for batch in train_loader.iter() {
            let output = model.forward(batch);
            let loss   = output.loss;

            // Record scalar loss before backward so we can log it.
            let loss_scalar: f64 = loss.clone().into_scalar() as f64;
            train_loss_sum += loss_scalar;
            train_batches  += 1;

            // Backpropagation + parameter update.
            let grads    = GradientsParams::from_grads(loss.backward(), &model);
            model        = optim.step(cfg.learning_rate, model, grads);
        }

        let avg_train_loss = train_loss_sum / train_batches.max(1) as f64;

        // ── Validation pass ───────────────────────────────────────────────────
        let mut valid_loss_sum = 0.0_f64;
        let mut valid_batches  = 0_usize;

        for batch in valid_loader.iter() {
            // Use the inner (non-autodiff) model for validation.
            let v_output = model.valid().forward(batch);
            let v_loss: f64 = v_output.loss.into_scalar() as f64;
            valid_loss_sum += v_loss;
            valid_batches  += 1;
        }

        let avg_valid_loss = if valid_batches > 0 {
            valid_loss_sum / valid_batches as f64
        } else {
            f64::NAN
        };

        let elapsed = t_start.elapsed().as_secs_f64();
        println!(
            "Epoch {:>3}/{} | train_loss: {:.4} | valid_loss: {:.4} | {:.1}s",
            epoch, cfg.num_epochs, avg_train_loss, avg_valid_loss, elapsed
        );

        // ── Checkpoint ────────────────────────────────────────────────────────
        let ckpt_path = output_dir.join(format!("checkpoint_epoch_{epoch:03}"));
        model
            .clone()
            .save_file(ckpt_path, &CompactRecorder::new())
            .expect("checkpoint save failed");
    }

    // ── 7. Save final model ───────────────────────────────────────────────────
    let final_path = output_dir.join("model_final");
    model
        .save_file(final_path.clone(), &CompactRecorder::new())
        .expect("final model save failed");

    println!("\n[train] Training complete. Final model saved to '{}'.", output_dir.display());
}
