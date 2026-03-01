# word-doc-qa

A command-line question-answering system that reads `.docx` documents and answers natural-language questions using a Transformer model trained with the [Burn](https://github.com/tracel-ai/burn) framework.

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Rust + Cargo | 1.75 or later | Install from [rustup.rs](https://rustup.rs) |
| GPU driver (optional) | — | WGPU backend falls back to CPU automatically |
| Microsoft Word `.docx` files | — | Place in a directory of your choice |

No Python, no pip, no conda — pure Rust.

---

## Getting the Code

```bash
git clone https://github.com/Skiet88/word-doc-qa-burn.git
cd word-doc-qa-burn
```

---

## Building

### Debug build (faster to compile, slower to run)
```bash
cargo build
```

### Release build (recommended)
```bash
cargo build --release
```

Output: `./target/release/word-doc-qa.exe` (Windows) or `./target/release/word-doc-qa` (Linux/macOS).

---

## Usage

```
word-doc-qa <mode> <docs_dir> [model_dir]
```

| Argument | Description |
|----------|-------------|
| `mode` | `train` or `infer` |
| `docs_dir` | Path to a directory containing `.docx` files |
| `model_dir` | *(optional)* Where to save/load model files. Defaults to `./model_output` |

---

## Running Inference (pre-trained model included)

A trained model is already present in `./model_output`. No training required.

```powershell
# Windows — release build
.\target\release\word-doc-qa.exe infer .\src\Documents

# Windows — debug build
.\target\debug\word-doc-qa.exe infer .\src\Documents
```

```bash
# Linux / macOS
./target/release/word-doc-qa infer ./src/Documents
```

The system loads all `.docx` files in the given directory and enters an interactive question loop:

```
[inference] Loaded 3 document(s), 207 paragraph(s).
Ask a question (or type 'exit' to quit):
> When is Freedom Day in 2026?
Answer: Monday 27 April 2026 - FREEDOM DAY

> How many board meetings does SARTEC Governance have in 2026?
SARTEC Governance had 4 board meeting(s) in 2026:
  1. Tuesday 31 March 2026  — SARETEC Governance Board Meeting (09:00)
  2. Tuesday 23 June 2026   — SARETEC Governance Board Meeting (09:00)
  3. Tuesday 27 October 2026 — SARETEC Governance Board Meeting (09:00)
  4. Tuesday 24 November 2026 — SARETEC Governance Board Meeting (09:00)

> exit
```

### Supported question types

| Type | Example |
|------|---------|
| Date / event lookup | `When is Heritage Day 2025?` |
| Term / semester dates | `When does the second semester start in 2026?` |
| Count — entity meetings | `How many times does the Senate meet in 2026?` |
| Count — month scoped | `How many public holidays are in April 2026?` |
| Acronym queries | `How many times did the HDC meet in 2025?` |

---

## Training from Scratch

To train on your own `.docx` documents:

```powershell
# Windows
.\target\release\word-doc-qa.exe train .\src\Documents .\model_output
```

```bash
# Linux / macOS
./target/release/word-doc-qa train ./src/Documents ./model_output
```

Default training settings:

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Backend | WGPU (GPU) with CPU fallback |

Checkpoints are saved each epoch as `model_output/checkpoint_epoch_NNN.mpk`.  
The final model is written to `model_output/model_final.mpk`.

> **Note:** Epoch checkpoints are excluded from the repository via `.gitignore` — they are large binary files not needed for inference.  
> Only `model_final.mpk`, `model_config.json`, and `vocab.json` are tracked by git.

---

## Project Structure

```
src/
  main.rs              # CLI entry point (train / infer modes)
  data/                # Document loading, tokenisation, dataset builder
  model/               # Transformer encoder + QA span-prediction head
  training/            # Training loop, optimiser, checkpointing
  inference/           # Retrieval, count/frequency queries, interactive loop
  Documents/           # Default location for .docx input files
model_output/
  model_final.mpk      # Trained weights  (required for inference)
  model_config.json    # Model architecture config
  vocab.json           # Tokeniser vocabulary
docs/
  report.md            # Full project report
```

---

## Troubleshooting

**`Error: model_output/model_final.mpk not found`**  
Run `train` first, or ensure the three required files (`model_final.mpk`, `model_config.json`, `vocab.json`) are present in `model_output/`.

**Cannot rebuild on Windows — binary is locked**  
Stop any running instance before building:
```powershell
Stop-Process -Name "word-doc-qa" -ErrorAction SilentlyContinue
```

**VS Code Source Control / git diff is slow**  
Old versions of the repo had the 50 epoch checkpoint files tracked. Remove them if present:
```bash
git rm --cached "model_output/checkpoint_epoch_*.mpk"
git commit -m "chore: untrack epoch checkpoints"
```

---

## License

MIT
