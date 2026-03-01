/// Burn Dataset implementation for extractive QA.
///
/// Each `QaItem` represents one training example:
///   input_ids    : [CLS] question [SEP] context [SEP]  (length = MAX_SEQ_LEN)
///   attn_mask    : 1 for real token, 0 for [PAD]
///   start_pos    : index of the first answer token in `input_ids`
///   end_pos      : index of the last  answer token in `input_ids`
use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};

use crate::data::tokenizer::{
    self, Vocab, attention_mask, find_answer_span,
};
use crate::data::loader::Document;

// The maximum total sequence length fed to the model.
pub const MAX_SEQ_LEN: usize = 256;

// ── Item ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaItem {
    /// Token IDs: [CLS] question tokens [SEP] context tokens [SEP] + padding
    pub input_ids: Vec<i64>,
    /// Attention mask (1 = real, 0 = pad)
    pub attn_mask: Vec<i64>,
    /// Token index of the answer start (inclusive)
    pub start_pos: i64,
    /// Token index of the answer end (inclusive)
    pub end_pos: i64,
}

// ── Dataset ───────────────────────────────────────────────────────────────────

pub struct QaDataset {
    items: Vec<QaItem>,
}

impl QaDataset {
    pub fn new(items: Vec<QaItem>) -> Self {
        Self { items }
    }
}

impl Dataset<QaItem> for QaDataset {
    fn get(&self, index: usize) -> Option<QaItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

// ── Synthetic Training-data Generator ────────────────────────────────────────

/// Build multiple (question, context, answer) triples from a paragraph.
///
/// Generates up to four question types so the model sees real-world patterns:
///   1. Date/event questions — when a month name appears in a sentence
///   2. Count questions       — when a digit appears in a sentence
///   3. General topic question — second-sentence key-words → first sentence
///   4. Fallback              — single-sentence: first 8 words as answer
///
/// All answers are sub-spans of the paragraph (extractive).
fn make_qa_triple(paragraph: &str) -> Vec<(String, String, String)> {
    let paragraph = paragraph.trim();
    if paragraph.len() < 30 {
        return vec![];
    }

    let mut triples: Vec<(String, String, String)> = Vec::new();
    let lower = paragraph.to_lowercase();

    // Split into sentences (keep delimiter so answers are complete phrases).
    let sentences: Vec<&str> = paragraph
        .split_inclusive(|c| c == '.' || c == '!' || c == '?')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        return triples;
    }

    // ── Strategy 1: Date / event questions ───────────────────────────────────
    // For every sentence that mentions a month, generate "When is <event>?"
    // and "What date is <event> held?" questions.
    const MONTHS: &[&str] = &[
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ];
    for &sent in &sentences {
        let sl = sent.to_lowercase();
        if let Some(month_byte_pos) = MONTHS.iter().filter_map(|m| sl.find(m)).min() {
            // Everything before the month name is the event / subject.
            let event = sent[..month_byte_pos]
                .trim()
                .trim_end_matches(|c: char| !c.is_alphanumeric());
            let date_part = sent[month_byte_pos..].trim();
            if event.len() > 4 && !date_part.is_empty() {
                triples.push((
                    format!("When is {}?", event),
                    paragraph.to_string(),
                    sent.to_string(),
                ));
                triples.push((
                    format!("What month and date is {} held?", event),
                    paragraph.to_string(),
                    sent.to_string(),
                ));
                triples.push((
                    format!("What is the date of {}?", event),
                    paragraph.to_string(),
                    sent.to_string(),
                ));
            }
        }
    }

    // ── Strategy 2: Count / number questions ─────────────────────────────────
    // For every sentence that contains a digit, generate varied count /
    // frequency / duration questions so the model sees real-world "How many",
    // "How often", and "How long" query patterns.
    for &sent in &sentences {
        let words: Vec<&str> = sent.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if word.chars().any(|c| c.is_ascii_digit()) {
                // The word immediately after the number is typically the unit
                // ("times", "days", "weeks", "meetings", …).
                let unit  = words.get(i + 1).copied().unwrap_or("items");
                // Two words before the number give subject context.
                let subj  = words.get(i.saturating_sub(2)).copied().unwrap_or("this");
                // Topic: first few non-trivial words of the sentence.
                let topic: String = words.iter()
                    .filter(|w| w.len() > 3)
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ");

                // "How many X are there?" — generic count
                triples.push((
                    format!("How many {} are there?", unit),
                    paragraph.to_string(),
                    sent.to_string(),
                ));
                // "How many times did … ?" — frequency pattern
                triples.push((
                    format!("How many times does {} occur?", subj),
                    paragraph.to_string(),
                    sent.to_string(),
                ));
                // "How often …?" — frequency synonym
                if !topic.is_empty() {
                    triples.push((
                        format!("How often does {}?", topic),
                        paragraph.to_string(),
                        sent.to_string(),
                    ));
                }
                // Duration pattern: "How long is …?" / "How many days/weeks …?"
                // Detect whether the unit following the number is a time unit
                // (singular or plural) — e.g. "day", "days", "week", "weeks".
                let unit_stem = unit.to_lowercase();
                let unit_stem = unit_stem.trim_end_matches('s'); // "days"→"day", "weeks"→"week"
                let is_duration_unit = matches!(
                    unit_stem,
                    "day" | "week" | "month" | "hour" | "minute" | "year" | "period"
                );
                if is_duration_unit {
                    triples.push((
                        format!("How long is {}?", topic),
                        paragraph.to_string(),
                        sent.to_string(),
                    ));
                    triples.push((
                        format!("How many {} does {} last?", unit, subj),
                        paragraph.to_string(),
                        sent.to_string(),
                    ));
                }
                break; // one question-cluster per sentence is enough
            }
        }
    }

    // ── Strategy 3: Duration / span questions from term-start/end paragraphs ─
    // When a paragraph mentions both a start and an end (or a recess range),
    // generate "How long is the X?" questions so the model is exposed to
    // duration queries during training.
    {
        let has_start = lower.contains("start of term") || lower.contains("start of year")
            || lower.contains("schools open") || lower.contains("begin");
        let has_end   = lower.contains("end of term") || lower.contains("schools close")
            || lower.contains("end of year");
        let has_weeks = lower.contains("week") || lower.contains("days") || lower.contains("month");

        if (has_start && has_end) || has_weeks {
            let subject: String = sentences[0]
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .take(4)
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !subject.is_empty() {
                triples.push((
                    format!("How long is {}?", subject),
                    paragraph.to_string(),
                    sentences[0].to_string(),
                ));
                triples.push((
                    format!("What is the duration of {}?", subject),
                    paragraph.to_string(),
                    sentences[0].to_string(),
                ));
            }
        }
    }

    // ── Strategy 4: General topic questions ──────────────────────────────────
    if sentences.len() >= 2 {
        let key_words: String = sentences[1]
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .take(5)
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !key_words.is_empty() {
            triples.push((
                format!("What is mentioned about {}?", key_words),
                paragraph.to_string(),
                sentences[0].to_string(),
            ));
        }
    } else {
        // ── Strategy 5: Single-sentence fallback ─────────────────────────────
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        let ans = words[..8.min(words.len())].join(" ");
        triples.push((
            "What is mentioned at the start of this passage?".to_string(),
            paragraph.to_string(),
            ans,
        ));
    }

    // Also add a "what does … say about" pair using the whole paragraph as
    // answer pointer so the model sees full-paragraph extractions too.
    if !lower.is_empty() {
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        let topic = words[..4.min(words.len())].join(" ");
        let ans_words = words[..10.min(words.len())].join(" ");
        if topic.len() > 5 {
            triples.push((
                format!("What does the document say about {}?", topic),
                paragraph.to_string(),
                ans_words,
            ));
        }
    }

    triples
}

/// Encode a single (question, context, answer) triple into a `QaItem`.
///
/// Layout of `input_ids`:
///   [CLS] q₁ q₂ … [SEP] c₁ c₂ … [SEP] [PAD] [PAD] …
fn encode_item(
    question: &str,
    context: &str,
    answer: &str,
    vocab: &Vocab,
) -> Option<QaItem> {
    // Tokenise separately for easier span search.
    let q_toks = tokenizer::tokenize(question);
    let a_toks = tokenizer::tokenize(answer);
    let c_toks = tokenizer::tokenize(context);

    // How many context tokens fit after [CLS] q [SEP] and the final [SEP]?
    // Layout: 1 + q_len + 1 + c_len + 1 ≤ MAX_SEQ_LEN
    let q_len = q_toks.len().min(MAX_SEQ_LEN / 4);
    let c_budget = MAX_SEQ_LEN.saturating_sub(q_len + 3);
    let c_len = c_toks.len().min(c_budget);

    // Build the combined token-id sequence.
    let mut ids: Vec<i64> = Vec::with_capacity(MAX_SEQ_LEN);
    ids.push(tokenizer::CLS_ID as i64);
    ids.extend(q_toks[..q_len].iter().map(|t| vocab.get(t) as i64));
    ids.push(tokenizer::SEP_ID as i64);

    let ctx_start = ids.len(); // first context token index
    ids.extend(c_toks[..c_len].iter().map(|t| vocab.get(t) as i64));
    ids.push(tokenizer::SEP_ID as i64);

    // Pad to MAX_SEQ_LEN.
    ids.resize(MAX_SEQ_LEN, tokenizer::PAD_ID as i64);

    let attn_mask = attention_mask(&ids);

    // Locate the answer span inside the context portion of the sequence.
    let a_ids: Vec<i64> = a_toks.iter().map(|t| vocab.get(t) as i64).collect();
    let ctx_slice = &ids[ctx_start..ctx_start + c_len];

    let (rel_start, rel_end) = find_answer_span(ctx_slice, &a_ids)?;
    let abs_start = (ctx_start + rel_start) as i64;
    let abs_end = (ctx_start + rel_end) as i64;

    // Sanity: positions must be within [0, MAX_SEQ_LEN-1].
    if abs_start >= MAX_SEQ_LEN as i64 || abs_end >= MAX_SEQ_LEN as i64 {
        return None;
    }

    Some(QaItem {
        input_ids: ids,
        attn_mask,
        start_pos: abs_start,
        end_pos: abs_end,
    })
}

/// Generate a `QaDataset` (and optional validation split) from a list of documents.
///
/// Returns `(train_dataset, valid_dataset)`.
pub fn build_datasets(
    docs: &[Document],
    vocab: &Vocab,
    valid_ratio: f64,
) -> (QaDataset, QaDataset) {
    let mut items: Vec<QaItem> = Vec::new();

    for doc in docs {
        for para in &doc.paragraphs {
            for (q, ctx, ans) in make_qa_triple(para) {
                if let Some(item) = encode_item(&q, &ctx, &ans, vocab) {
                    items.push(item);
                }
            }
        }
    }

    // Shuffle deterministically so results are reproducible.
    // Simple in-place Fisher-Yates with a fixed seed.
    let n = items.len();
    if n == 0 {
        return (QaDataset::new(vec![]), QaDataset::new(vec![]));
    }

    let mut seed: u64 = 42;
    for i in (1..n).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        items.swap(i, j);
    }

    let split = ((n as f64) * (1.0 - valid_ratio)) as usize;
    let split_idx = split.max(1).min(n - 1);
    let valid_items = items.split_off(split_idx);
    (QaDataset::new(items), QaDataset::new(valid_items))
}

// ── Batcher ───────────────────────────────────────────────────────────────────

use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

/// A batch of QA examples ready for the model.
#[derive(Debug, Clone)]
pub struct QaBatch<B: Backend> {
    /// [batch, MAX_SEQ_LEN]  integer token IDs
    pub input_ids: Tensor<B, 2, Int>,
    /// [batch, MAX_SEQ_LEN]  1/0 attention mask
    pub attn_mask: Tensor<B, 2, Int>,
    /// [batch]  start token index
    pub start_positions: Tensor<B, 1, Int>,
    /// [batch]  end token index
    pub end_positions: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct QaBatcher<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> QaBatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self { _backend: std::marker::PhantomData }
    }
}

impl<B: Backend> Batcher<B, QaItem, QaBatch<B>> for QaBatcher<B> {
    fn batch(&self, items: Vec<QaItem>, device: &B::Device) -> QaBatch<B> {
        let batch = items.len();
        let seq = MAX_SEQ_LEN;

        let ids_flat: Vec<i64>  = items.iter().flat_map(|i| i.input_ids.iter().copied()).collect();
        let mask_flat: Vec<i64> = items.iter().flat_map(|i| i.attn_mask.iter().copied()).collect();
        let starts: Vec<i64>    = items.iter().map(|i| i.start_pos).collect();
        let ends: Vec<i64>      = items.iter().map(|i| i.end_pos).collect();

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids_flat, [batch, seq]),
            device,
        );
        let attn_mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_flat, [batch, seq]),
            device,
        );
        let start_positions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(starts, [batch]),
            device,
        );
        let end_positions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ends, [batch]),
            device,
        );

        QaBatch { input_ids, attn_mask, start_positions, end_positions }
    }
}
