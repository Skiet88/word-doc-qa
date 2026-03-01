/// Word-level vocabulary and tokenizer built from the training corpus.
///
/// Uses simple whitespace + punctuation splitting so there are no external
/// model files required beyond the documents themselves.
use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

// ── Special token IDs (must stay stable) ─────────────────────────────────────
pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const CLS_ID: usize = 2;
pub const SEP_ID: usize = 3;

const SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]"];

// ── Vocabulary ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

impl Vocab {
    pub fn new() -> Self {
        let mut v = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };
        for tok in SPECIAL_TOKENS {
            v.add(tok);
        }
        v
    }

    /// Insert `token` if absent; return its ID in either case.
    pub fn add(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len();
        self.id_to_token.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    /// Look up a token; returns [UNK] ID if not found.
    pub fn get(&self, token: &str) -> usize {
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

// ── Tokenisation helpers ──────────────────────────────────────────────────────

/// Split text into lowercase word tokens on whitespace and ASCII punctuation
/// (apostrophes are kept inside tokens, e.g. "don't").
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Build a vocabulary from a list of text strings, capped at `max_vocab` entries.
/// Special tokens are always included.
pub fn build_vocab(texts: &[String], max_vocab: usize) -> Vocab {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for tok in tokenize(text) {
            *freq.entry(tok).or_insert(0) += 1;
        }
    }

    // Sort by descending frequency; break ties lexicographically.
    let mut sorted: Vec<(String, usize)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut vocab = Vocab::new();
    let budget = max_vocab.saturating_sub(SPECIAL_TOKENS.len());
    for (token, _) in sorted.into_iter().take(budget) {
        vocab.add(&token);
    }
    vocab
}

/// Build the attention mask for an encoded sequence:
/// 1 for real tokens, 0 for [PAD].
pub fn attention_mask(ids: &[i64]) -> Vec<i64> {
    ids.iter()
        .map(|&id| if id == PAD_ID as i64 { 0 } else { 1 })
        .collect()
}

/// Find the first position in `context_ids` (a full [CLS]+question+[SEP]+ctx+[SEP] sequence)
/// where `answer_ids` appears.  Returns `(start, end)` token indices (inclusive), or `None`.
pub fn find_answer_span(context_ids: &[i64], answer_ids: &[i64]) -> Option<(usize, usize)> {
    if answer_ids.is_empty() {
        return None;
    }
    for start in 0..context_ids.len().saturating_sub(answer_ids.len() - 1) {
        if context_ids[start..start + answer_ids.len()] == answer_ids[..] {
            return Some((start, start + answer_ids.len() - 1));
        }
    }
    None
}
