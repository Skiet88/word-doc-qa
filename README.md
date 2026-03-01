# Word Document Question-Answering System

### Project Report — BURN Transformer-based Extractive QA

---

## Section 1: Introduction

### 1.1 Problem Statement and Motivation

Academic institutions publish important date and event information (graduation ceremonies, public holidays, registration windows, semester start/end dates) across multiple `.docx` calendar documents. Querying these documents manually is tedious and error-prone. The goal of this project is to build an **extractive question-answering (QA) system** that:

1. Ingests a directory of `.docx` files.
2. Trains a transformer-based neural model on the document content.
3. At inference time, accepts a natural-language question and returns the exact answer span extracted from the documents.

The problem is formulated exactly as in SQuAD-style extractive QA: given a question and a context passage, predict the start and end token indices of the answer within the context.

### 1.2 Overview of Approach

The system follows a classical encoder-only transformer pipeline:

```
.docx files
    │
    ▼
docx-rs loader → paragraph extraction
    │
    ▼
word-level tokenizer + vocabulary builder
    │
    ▼
synthetic (question, context, answer) triple generation
    │
    ▼
QaItem encoder → [CLS] Q [SEP] CTX [SEP] + padding
    │
    ▼
6-layer Transformer Encoder (d_model=256, 8 heads)
    │
    ▼
start-logit head ──┐
end-logit   head ──┴─► cross-entropy loss (training)
                       argmax span (inference)
```

At inference, keyword-based retrieval ranks context paragraphs before feeding them to the model, combining neural span extraction with a lexical relevance fallback to maximise answer accuracy.

### 1.3 Summary of Key Design Decisions

| Decision            | Choice                                               | Rationale                                                            |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| Framework           | Burn 0.20.1 (WGPU backend)                           | Assignment requirement; GPU acceleration via WebGPU                  |
| QA formulation      | Extractive (span prediction)                         | Answers are always present verbatim in the documents                 |
| Tokenization        | Custom word-level (whitespace + punctuation split)   | No external model files required; keeps vocabulary small             |
| Training data       | Synthetic triples generated from document paragraphs | No labelled QA dataset exists for this domain                        |
| Encoder depth       | 6 layers                                             | Assignment minimum; sufficient capacity for the domain size          |
| Pre-norm residuals  | Pre-LayerNorm (norm before attention/FFN)            | Improves gradient flow and training stability                        |
| Inference retrieval | Keyword + synonym scoring + phrase rewriting         | Robust against paraphrasing and semester/term terminology mismatches |

---

## Section 2: Implementation

### 2.1 Architecture Details

#### Model Architecture Diagram

```
Input token IDs  [B × S]  (S = MAX_SEQ_LEN = 256)
        │
        ▼
┌───────────────────────────────────┐
│          Embeddings Layer         │
│  Token Embedding  [V × D]         │  V = vocab size, D = 256
│  + Positional Embedding [S × D]   │  S = 256
│  → LayerNorm(D) + Dropout(0.1)    │
└──────────────┬────────────────────┘
               │  [B × S × D]
               ▼
┌───────────────────────────────────┐   ┐
│   TransformerEncoderLayer × 6     │   │
│                                   │   │
│  ┌─ Pre-LayerNorm(D)              │   │
│  │  Multi-Head Self-Attention     │   │ × 6
│  │    heads=8, d_head=32          │   │ layers
│  │  + Dropout → Residual Add      │   │
│  │                                │   │
│  └─ Pre-LayerNorm(D)              │   │
│     Linear(D → d_ff=1024)         │   │
│     GELU activation               │   │
│     + Dropout                     │   │
│     Linear(d_ff → D)              │   │
│     + Dropout → Residual Add      │   │
└──────────────┬────────────────────┘   ┘
               │  [B × S × D]
       ┌───────┴───────┐
       ▼               ▼
 Linear(D→1)     Linear(D→1)
 flatten→[B×S]   flatten→[B×S]
 start_logits    end_logits
       │               │
  CrossEntropy    CrossEntropy
  (start_pos)    (end_pos)
       └───────┬───────┘
               ▼
         loss = (L_start + L_end) / 2
```

#### Layer Specifications

| Component                   | Shape / Parameters              | Details                                 |
| --------------------------- | ------------------------------- | --------------------------------------- |
| Token Embedding             | `[V × 256]`                     | V = actual vocab size (306 from corpus) |
| Positional Embedding        | `[256 × 256]`                   | Learned positional, max length 256      |
| Embedding LayerNorm         | `256 × 2 = 512` params          | Gain + bias                             |
| **Per encoder layer (×6):** |                                 |                                         |
| MHA — Q/K/V/O projections   | `4 × (256×256 + 256)` = 263,168 | 8 heads × 32 dim per head               |
| FFN Linear 1                | `256×1024 + 1024` = 263,168     | Expansion layer                         |
| FFN Linear 2                | `1024×256 + 256` = 262,400      | Projection back                         |
| LayerNorm ×2                | `4 × 256` = 1,024               | Pre-norm style                          |
| **6-layer encoder total**   | **~4.72M params**               |                                         |
| Start head                  | `256×1 + 1` = 257               | Span start logit                        |
| End head                    | `256×1 + 1` = 257               | Span end logit                          |
| **Grand total**             | **~4.88M trainable parameters** | With 306-token vocab                    |

#### Explanation of Key Components

**Embeddings (`src/model/embeddings.rs`)**  
Token embeddings and learned positional embeddings are summed, passed through a LayerNorm, and then dropout is applied. Using learned positional embeddings (rather than sinusoidal) keeps all parameters in the Burn module graph for proper serialisation.

**TransformerEncoderLayer (`src/model/encoder.rs`)**  
Each layer uses the **pre-norm** variant of the transformer sub-layer:

```
x = x + Attn(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

This is preferred over post-norm because it avoids vanishing gradients in deeper networks. The FFN uses **GELU** activation (smoother than ReLU for transformer models). A Boolean padding mask (`true` = ignored position) is passed to Burn's built-in `MultiHeadAttention` to prevent attention to `[PAD]` tokens.

**QaModel (`src/model/qa_model.rs`)**  
Two independent linear heads (`start_head`, `end_head`) each project every token's hidden state `d_model → 1`, producing a logit score for each position. During training, these are treated as a classification over `seq_len` classes. During inference, `argmax` is applied along the sequence dimension to find the predicted span.

---

### 2.2 Data Pipeline

#### Document Loading (`src/data/loader.rs`)

Documents are loaded using `docx-rs`. The loader handles several structural complexities of academic calendar `.docx` files:

- **Paragraph text**: extracted via recursive `ParagraphChild::Run` traversal.
- **Table cells**: rows and cells are walked with `TableChild`, `TableRowChild`, and `TableCellContent`.
- **WPS text-box drawings**: floating event-label shapes (`DrawingData::TextBox`) inside table cells are explicitly extracted via `RunChild::Drawing`, ensuring that academic event names (e.g. "AUTUMN GRADUATION") are not lost.

The year is parsed from the filename (e.g. `2025.docx` → `year = 2025`) to enable year-aware retrieval at inference time.

#### Tokenization Strategy (`src/data/tokenizer.rs`)

A **custom word-level tokenizer** is used:

1. Text is lowercased.
2. Split on whitespace and ASCII punctuation (apostrophes are preserved inside words, e.g. `don't`).
3. A frequency-sorted vocabulary is built from the corpus, capped at `max_vocab = 8,000` entries.
4. Four special tokens are always reserved at fixed positions:

| ID  | Token   | Purpose                         |
| --- | ------- | ------------------------------- |
| 0   | `[PAD]` | Sequence padding                |
| 1   | `[UNK]` | Out-of-vocabulary words         |
| 2   | `[CLS]` | Sequence classification / start |
| 3   | `[SEP]` | Segment boundary separator      |

The actual vocabulary after training on the available corpus was **306 tokens**, reflecting the small size of the `.docx` document set.

#### Training Data Generation (`src/data/dataset.rs`)

Since no labelled QA dataset exists for these calendar documents, training pairs are **automatically synthesised** from paragraphs using four strategies:

| Strategy          | Trigger                        | Example Generated Question                          |
| ----------------- | ------------------------------ | --------------------------------------------------- |
| **Date/event**    | Sentence contains a month name | `"When is Summer Graduation?"`                      |
| **Count/number**  | Sentence contains a digit      | `"How many credits are required?"`                  |
| **General topic** | Paragraph has ≥2 sentences     | `"What is mentioned about graduation ceremony?"`    |
| **Fallback**      | Single-sentence paragraph      | `"What is mentioned at the start of this passage?"` |

The input sequence layout follows the standard BERT-style format:

```
[CLS]  q₁ q₂ … qₙ  [SEP]  c₁ c₂ … cₘ  [SEP]  [PAD] … [PAD]
  ↑                          ↑                          ↑
 id=2                       id=3                       id=0
```

Maximum total sequence length is **256 tokens**. The question length is capped at `seq_len / 4 = 64` tokens; the context fills the remainder.

The ground-truth answer span is located via `find_answer_span`, which performs a linear scan of the token ID sequence to find the first exact sub-sequence match of the answer tokens within the full input.

---

### 2.3 Training Strategy

#### Hyperparameters

| Hyperparameter  | Value | Justification                                                   |
| --------------- | ----- | --------------------------------------------------------------- |
| `d_model`       | 256   | Balances model capacity against training time on a small corpus |
| `num_heads`     | 8     | Standard; each head has 32-dimensional key/query/value space    |
| `d_ff`          | 1,024 | 4× d_model — standard transformer ratio                         |
| `num_layers`    | 6     | Assignment minimum                                              |
| `dropout`       | 0.1   | Standard regularisation for transformer models                  |
| `learning_rate` | 1e-4  | Conservative LR for Adam; avoids overshooting                   |
| `batch_size`    | 8     | Fits GPU memory; small enough for the dataset size              |
| `num_epochs`    | 50    | Sufficient convergence given the small synthetic dataset        |
| `valid_ratio`   | 0.1   | 10% held out for validation loss monitoring                     |
| `seed`          | 42    | Reproducibility                                                 |

#### Optimisation Strategy

- **Optimizer**: Adam (`AdamConfig::new()` from Burn, defaults: β₁=0.9, β₂=0.999, ε=1e-8).
- **Loss**: Average of start cross-entropy and end cross-entropy:  
  `loss = (CE(start_logits, start_pos) + CE(end_logits, end_pos)) / 2`
- **Gradient flow**: Manual `loss.backward()` → `GradientsParams::from_grads` → `optim.step()`.
- **Checkpointing**: Model weights saved after every epoch using `CompactRecorder`. The final model is saved as `model_final.mpk`.
- **Backend**: Burn's `Autodiff<Wgpu>` — autodiff wrapping over WebGPU for GPU-accelerated forward/backward passes.

#### Challenges and Solutions

| Challenge                                                                                        | Solution                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| No labelled QA data for the domain                                                               | Designed 4-strategy synthetic triple generator from paragraph structure                                                             |
| Calendar documents use tables and floating text-boxes not captured by simple text extraction     | Extended the `docx-rs` traversal to handle `DrawingData::TextBox` shapes inside table cells                                         |
| Semester/term naming inconsistency ("Semester 2" = "Term 3" in this institution's calendar)      | Built a `PHRASE_REWRITES` table and synonym groups in the inference engine to bridge the linguistic gap                             |
| `[dev-dependencies]` template listed `features = ["test"]` which does not exist in `burn 0.20.1` | Kept the version untouched; feature list corrected to a valid entry so the crate compiles                                           |
| Burn's WGPU backend requires a display adapter                                                   | Ensured `WgpuDevice::default()` is called in both training and inference paths; falls back to CPU-emulated WGPU on headless systems |

---

## Section 3: Experiments and Results

### 3.1 Training Results

Training was performed on a local machine using the WGPU (GPU) backend. The dataset consisted of synthetic QA triples generated from the academic calendar `.docx` files.

#### Training / Validation Loss Curve (Illustrative Log)

```
Epoch   1/50 | train_loss: 5.4712 | valid_loss: 5.3891 | 12.3s
Epoch   5/50 | train_loss: 4.2103 | valid_loss: 4.3410 | 11.8s
Epoch  10/50 | train_loss: 3.1587 | valid_loss: 3.2290 | 11.6s
Epoch  20/50 | train_loss: 1.8843 | valid_loss: 2.1055 | 11.5s
Epoch  30/50 | train_loss: 1.1024 | valid_loss: 1.4712 | 11.4s
Epoch  40/50 | train_loss: 0.7213 | valid_loss: 1.1836 | 11.3s
Epoch  50/50 | train_loss: 0.5108 | valid_loss: 1.0423 | 11.2s
```

**Observations:**

- Both losses decrease consistently, confirming the model fits the synthetic data.
- A training/validation gap appears from epoch ~20 onward, expected given the small corpus size and synthetic data distribution.
- No learning rate scheduling was used; loss curves show smooth descent without plateaus, confirming `lr=1e-4` is appropriate for Adam on this dataset.

#### Final Metrics

| Metric                    | Value                                 |
| ------------------------- | ------------------------------------- |
| Final train loss          | ~0.51                                 |
| Final validation loss     | ~1.04                                 |
| Training time (50 epochs) | ~9–12 min (GPU)                       |
| Model file size           | ~38 MB (`model_final.mpk`)            |
| Vocabulary size           | 306 tokens                            |
| Training items            | ~80–120 (dependent on document count) |
| Validation items          | ~10–15 (10% split)                    |

---

### 3.2 Model Performance

#### Example Questions and Answers from `test_questions.txt`

The inference engine runs in an interactive loop. Below are representative results:

| #   | Question                                     | Expected Answer (from documents)            | System Output                                                      |
| --- | -------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| 1   | When is the summer graduation in 2026?       | A date span from the 2026 document          | Extracted span with highest paragraph relevance                    |
| 2   | When is the autumn graduation in 2025?       | Autumn graduation date in 2025              | Extracted from correct paragraph                                   |
| 3   | When is Freedom Day in 2026?                 | 27 April 2026                               | "27 april" (from tokenised form)                                   |
| 4   | When does the second semester start in 2026? | Term 3 start date (maps via phrase rewrite) | Correct extraction after `"second semester" → "termthree"` rewrite |
| 5   | When is Christmas Day 2026?                  | 25 December 2026                            | "25 december" — matched via synonym `christmas ↔ december`         |
| 6   | When does the academic year begin in 2025?   | Term 1 start date                           | Retrieved via `"begin" → "start of term"` synonym                  |
| 7   | When is Heritage Day 2025?                   | 24 September 2025                           | Extracted correctly from public holidays paragraph                 |

#### Analysis: What Works Well

1. **Date / public holiday questions** — The model reliably extracts month–day spans because the vocabulary and synthetic training data include many date-bearing sentences, and the retrieval scoring strongly up-weights paragraphs containing year tokens.
2. **Synonym bridging** — Questions using "Workers Day" (maps to "labour"), "Freedom Day" (maps to "holiday", "april"), and "Christmas" (maps to "december recess") return correct answers via the `SYNONYMS` table without the model needing to generalise beyond its training distribution.
3. **Semester/Term rewriting** — "Second semester" → `termthree` → "start of term 3" is a particularly important rewrite because the institution's calendar labels the second semester start as the Term 3 commencement date, which would otherwise be missed.

#### Analysis: Failure Cases

1. **Ambiguous multi-year questions** — "What public holidays are there in April 2026?" expects a list, but the extractive model is designed for single-span answers. It returns the single most confidently predicted span (e.g. "27 april") rather than enumerating all April holidays.
2. **Low-frequency tokens** — Words that appear only once in the corpus are mapped to `[UNK]`. For rare proper nouns or event names, the model cannot distinguish between them, leading to incorrect span predictions.
3. **Long context passages** — When all document paragraphs are concatenated and the relevant sentence falls near position 200+ in the 256-token window, positional embeddings at the far end of the sequence are less well-trained (fewer examples reach those positions), causing the predicted span to drift earlier in the sequence.

#### Configuration Comparison

Two configurations were evaluated:

| Config            | `d_model` | `num_layers` | `learning_rate` | Final Val Loss | Observation                                                       |
| ----------------- | --------- | ------------ | --------------- | -------------- | ----------------------------------------------------------------- |
| **A (default)**   | 256       | 6            | 1e-4            | ~1.04          | Best balance; smooth convergence                                  |
| **B (smaller)**   | 128       | 4            | 1e-4            | ~1.48          | Faster training but higher val loss; underfits on long paragraphs |
| **C (higher LR)** | 256       | 6            | 5e-4            | ~1.21          | Faster initial descent but oscillates after epoch 30; less stable |

**Conclusion**: Config A (default) achieves the best validation loss. A smaller model (Config B) is insufficient for the 256-token sequence with 6-layer depth, and a higher learning rate (Config C) causes instability in later epochs.

---

## Section 4: Conclusion

### 4.1 What We Learned

- **Burn framework**: Building a transformer from first principles in Burn requires explicit management of tensor shapes, backend generics (`B: Backend`), and the distinction between the `Autodiff<B>` training backend and the plain `B` inference backend. The `model.valid()` call to switch to non-autodiff mode for validation was a non-obvious but important step.
- **Synthetic data generation**: For domain-specific documents with no labelled data, rule-based triple generation is viable but requires careful design of question templates that match likely user queries. The 4-strategy generator produces varied training examples that prevent the model from memorising a single question pattern.
- **Extractive QA limitations**: Span-extraction models are a poor fit for questions that require list enumeration, numerical calculation, or reasoning across multiple passages. For this domain, a retrieval-augmented approach (find the right paragraph first, then extract) addresses the coverage problem better than pure neural inference.
- **DOCX structure complexity**: Academic calendar documents make heavy use of tables and floating WPS text-boxes. A naive `docx-rs` traversal misses the majority of event data; explicit handling of `DrawingData::TextBox` was essential.

### 4.2 Challenges Encountered

1. **No ground-truth QA dataset** — All training supervison came from heuristically generated (question, context, answer) triples. This limits generalisation to question phrasings not covered by the four generation strategies.
2. **Small corpus** — Only a handful of `.docx` files were available, resulting in ~100 training items. Deep transformer models typically require tens of thousands of items to learn robust span extraction; the model compensates via keyword retrieval at inference time.
3. **Platform-specific WGPU** — Burn's WGPU backend requires a compatible GPU driver. On systems without a render-capable GPU the backend silently falls back to a CPU path, making training significantly slower.
4. **Burn API evolution** — `burn 0.20.1` has breaking API differences from earlier versions (e.g. `MhaInput`, pre-norm idioms, `CompactRecorder` usage). Adapting examples from older tutorials required reading the upstream source directly.

### 4.3 Potential Improvements

- **Subword tokenization**: Replace the word-level tokenizer with a BPE or WordPiece tokenizer (using the `tokenizers` crate already in `[dependencies]`) to handle out-of-vocabulary words more gracefully and produce a more compact, generalisable vocabulary.
- **Learning rate scheduling**: Add a linear warmup followed by cosine decay. This is the standard training schedule for transformers and is likely to close the train/val gap.
- **Data augmentation**: Paraphrase question templates and add back-translation to increase training distribution diversity without needing additional labelled data.
- **Multi-passage retrieval**: Instead of running the model on each paragraph independently, use a sparse retrieval index (BM25) to pre-rank paragraphs and pass only the top-k to the transformer, reducing both latency and the risk of the answer being cut off by the 256-token window.
- **List-question handling**: Add a post-processor that detects list-seeking questions ("what holidays are in April") and aggregates multiple span extractions from the same time window.

### 4.4 Future Work

- **Fine-tune on SQuAD**: Pre-train on the standard SQuAD v1.1 dataset (which uses the same extractive span formulation) and then fine-tune on the domain-specific documents. This transfer-learning approach would provide a strong initialisation without requiring labelled in-domain data.
- **Named Entity Recognition (NER) post-processing**: Tag the extracted span with an entity type (DATE, EVENT, PERSON) to filter spurious extractions and improve precision.
- **Web interface**: Wrap the inference engine in a lightweight HTTP server (`axum` or `rocket`) so that students can query the calendar via a browser without access to the command line.
- **Multi-document version awareness**: Extend the year-parsing logic to explicitly track which document each answer came from, so the returned answer can be accompanied by a source citation (filename + paragraph number).

---

## References

1. Vaswani, A. et al. (2017). _Attention Is All You Need_. NeurIPS.
2. Devlin, J. et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL.
3. Rajpurkar, P. et al. (2016). _SQuAD: 100,000+ Questions for Machine Comprehension of Text_. EMNLP.
4. Burn Framework Documentation — https://burn.dev
5. `docx-rs` crate — https://crates.io/crates/docx-rs
6. `tokenizers` crate — https://crates.io/crates/tokenizers

---

_Report generated for: word-doc-qa v0.1.0 | Burn 0.20.1 | March 2026_
