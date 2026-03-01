/// Inference engine: load a trained model and answer questions about documents.
use std::io::{self, Write};
use std::path::Path;

use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    prelude::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, TensorData},
};

use crate::data::{
    dataset::MAX_SEQ_LEN,
    loader::load_all_docx,
    tokenizer::{attention_mask, tokenize, Vocab, CLS_ID, PAD_ID, SEP_ID},
};
use crate::model::qa_model::{QaModel, QaModelConfig};

type InferBackend = Wgpu;

// ── Model loading ─────────────────────────────────────────────────────────────

/// Load the final model and vocabulary from `model_dir`.
pub fn load_model(model_dir: &Path) -> (QaModel<InferBackend>, Vocab) {
    let device = WgpuDevice::default();

    // Load vocabulary.
    let vocab = Vocab::load(&model_dir.join("vocab.json"))
        .expect("vocab.json not found in model directory");

    // Re-instantiate model architecture from saved config.
    let cfg_str = std::fs::read_to_string(model_dir.join("model_config.json"))
        .expect("model_config.json not found");
    let model_cfg: QaModelConfig =
        serde_json::from_str(&cfg_str).expect("invalid model config JSON");

    let record = CompactRecorder::new()
        .load(model_dir.join("model_final").into(), &device)
        .expect("model_final not found — have you run `train` first?");

    let model = model_cfg.init::<InferBackend>(&device).load_record(record);

    (model, vocab)
}

// ── Answer extraction ─────────────────────────────────────────────────────────

/// Encode a single (question, context) pair the same way as the dataset batcher,
/// returning (input_ids, attn_mask, context_token_offset, context_tokens).
fn encode_qa_pair(
    question: &str,
    context: &str,
    vocab: &Vocab,
) -> (Vec<i64>, Vec<i64>, usize, Vec<String>) {
    let q_toks = tokenize(question);
    let c_toks = tokenize(context);

    let q_len = q_toks.len().min(MAX_SEQ_LEN / 4);
    let c_budget = MAX_SEQ_LEN.saturating_sub(q_len + 3);
    let c_len = c_toks.len().min(c_budget);

    let mut ids: Vec<i64> = Vec::with_capacity(MAX_SEQ_LEN);
    ids.push(CLS_ID as i64);
    ids.extend(q_toks[..q_len].iter().map(|t| vocab.get(t) as i64));
    ids.push(SEP_ID as i64);

    let ctx_offset = ids.len(); // first context token index
    ids.extend(c_toks[..c_len].iter().map(|t| vocab.get(t) as i64));
    ids.push(SEP_ID as i64);
    ids.resize(MAX_SEQ_LEN, PAD_ID as i64);

    let mask = attention_mask(&ids);
    let ctx_tokens = c_toks[..c_len].to_vec();

    (ids, mask, ctx_offset, ctx_tokens)
}

/// Run inference for `question` against `context`, returning the extracted
/// answer string and the keyword relevance `score` (higher = more confident).
/// `relevance` is forwarded unchanged so callers can rank candidates.
/// Returns `None` if the model's predicted span falls outside the context.
pub fn answer_question(
    model: &QaModel<InferBackend>,
    vocab: &Vocab,
    question: &str,
    context: &str,
    relevance: f32,
) -> Option<(String, f32)> {
    let device = WgpuDevice::default();
    let (ids, mask, ctx_offset, ctx_tokens) = encode_qa_pair(question, context, vocab);

    let seq = MAX_SEQ_LEN;

    let input_ids = Tensor::<InferBackend, 2, Int>::from_data(
        TensorData::new(ids.clone(), [1, seq]),
        &device,
    );
    let attn_mask = Tensor::<InferBackend, 2, Int>::from_data(
        TensorData::new(mask, [1, seq]),
        &device,
    );

    let predictions = model.predict(input_ids, attn_mask);
    let (start_tok, end_tok) = predictions[0];

    // Predicted span must land inside the context portion of the sequence.
    if start_tok < ctx_offset || start_tok >= ctx_offset + ctx_tokens.len() {
        return None;
    }
    let rel_start = start_tok - ctx_offset;
    let rel_end   = (end_tok.saturating_sub(ctx_offset)).min(ctx_tokens.len() - 1);
    let rel_end   = rel_end.max(rel_start);

    let answer_tokens = &ctx_tokens[rel_start..=rel_end];
    let answer = answer_tokens.join(" ");

    Some((answer, relevance))
}

// ── Keyword retrieval helpers ─────────────────────────────────────────────────

/// English stop-words that carry no discriminative value for retrieval.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "if", "with", "by", "from", "this",
    "that", "it", "its", "how", "what", "when", "where", "who", "which",
    "many", "much", "some", "any", "all", "their", "there", "here",
    "about", "into", "than", "then", "also", "each", "per",
];

/// Synonym/phrase groups: any question word that matches a group key will also
/// be credited when the paragraph contains any word in that group's value list.
const SYNONYMS: &[(&str, &[&str])] = &[
    ("graduation",   &["graduation", "graduate", "graduates", "ceremony", "commencement", "summer graduation", "autumn graduation"]),
    ("ceremony",     &["ceremony", "graduation", "commencement"]),
    ("summer",       &["summer", "december", "summer graduation"]),
    // "year" in a graduation context maps to the summer/end-of-year ceremony.
    ("year",         &["year", "summer", "december", "annual"]),
    ("end",          &["end", "summer", "december", "final"]),
    ("autumn",       &["autumn", "fall", "april", "autumn graduation"]),
    ("workers",      &["workers", "labour", "labor"]),
    ("labour",       &["labour", "workers", "labor"]),
    ("holiday",      &["holiday", "recess", "public"]),
    ("exam",         &["exam", "examination", "assessment", "assessments", "examination question"]),
    ("examination",  &["exam", "examination", "assessment", "assessments"]),
    ("registration", &["registration", "enrolment", "enrollment", "start of term", "wced schools open"]),
    ("enrolment",    &["registration", "enrolment", "enrollment"]),
    // Semester / term mapping
    // "semester" alone is kept broad; specific semester numbers are handled
    // via phrase rewrites that inject word-form tokens (termthree, termone, …)
    // before these synonyms are evaluated.
    ("semester",     &["semester", "start of term", "term"]),
    ("second",       &["second", "term 3", "term3", "semester 2", "start of term 3"]),
    ("first",        &["first", "term 1", "term1", "semester 1", "start of term 1"]),
    ("third",        &["third", "term 3", "start of term 3"]),
    ("fourth",       &["fourth", "term 4", "start of term 4"]),
    // Word-form tokens injected by normalize_question for "second semester" etc.
    // These avoid digit-substring false positives ("3" matching "13").
    ("termthree",    &["termthree", "term 3", "term3", "start of term 3"]),
    ("termfour",     &["termfour",  "term 4", "term4", "start of term 4"]),
    ("termone",      &["termone",   "term 1", "term1", "start of term 1"]),
    ("termtwo",      &["termtwo",   "term 2", "term2", "start of term 2"]),
    ("termstart",    &["termstart", "start of term"]),
    // Recess / vacation
    ("recess",       &["recess", "schools close", "wced schools close", "university holiday"]),
    ("winter",       &["winter", "june", "july"]),
    ("spring",       &["spring", "september", "october"]),
    // Classes / start of term
    ("classes",      &["classes", "start of term", "schools open"]),
    ("start",        &["start", "start of term", "first day", "begin"]),
    ("begin",        &["begin", "start of term", "first day"]),
    ("open",         &["open", "schools open", "start of term"]),
    ("close",        &["close", "schools close", "end of term"]),
    // Christmas / December recess
    ("christmas",    &["christmas", "december", "december recess"]),
    ("december",     &["december", "christmas"]),
    // Easter
    ("easter",       &["easter", "good friday", "family day", "easter sunday", "easter monday"]),
    ("good",         &["good", "good friday", "easter"]),
    ("friday",       &["friday", "good friday"]),
    // Public holidays general
    ("public",       &["public", "holiday", "day"]),
    ("june",         &["june", "winter", "youth day"]),
];

// ── Academic calendar phrase normalisation ────────────────────────────────────
//
// A "semester" at this university consists of two "terms":
//   Semester 1  =  Term 1  +  Term 2
//   Semester 2  =  Term 3  +  Term 4
//
// The second semester *starts* at the beginning of Term 3.
// This table rewrites multi-word question phrases so the downstream
// keyword-scorer sees the canonical term/event labels that appear in the
// calendar documents.  Replacements are applied left-to-right; longer
// phrases are listed before shorter ones so they take priority.
const PHRASE_REWRITES: &[(&str, &str)] = &[
    // ── Semester ↔ Term mapping ──────────────────────────────────────────────
    // A year has 2 semesters; each semester contains 2 terms:
    //   Semester 1 = Term 1 + Term 2   (starts at beginning of Term 1)
    //   Semester 2 = Term 3 + Term 4   (starts at beginning of Term 3)
    //
    // Word-form numbers are used ("three", "one") to avoid digit-in-substring
    // false positives (e.g., "3" falsely matching "13" in a date cell).
    ("second semester",    "semester termthree third termstart"),
    ("semester 2",         "semester termthree third termstart"),
    ("first semester",     "semester termone first termstart"),
    ("semester 1",         "semester termone first termstart"),
    // ── Term start / end labels ──
    ("start of semester",  "start of term"),
    ("end of semester",    "end of term"),
    ("semester start",     "start of term"),
    ("semester end",       "end of term"),
];

/// Rewrite a raw question string so that multi-word academic-calendar phrases
/// are replaced with tokens that map unambiguously to the correct calendar term.
///
/// Word-form tokens (e.g. `termthree`) are used instead of digit strings so
/// that the scorer's `contains()` check doesn't pick up digit substrings in
/// day numbers such as "13" containing "3".
///
/// Example:
///   `"When does the second semester start in 2026?"`
///   → `"when does the semester termthree third termstart start in 2026?"`
///
/// `termthree` maps (via SYNONYMS) to `"term 3"` / `"start of term 3"`,
/// so Term-3 paragraphs score significantly higher than Term-4 paragraphs.
fn normalize_question(question: &str) -> String {
    let mut result = question.to_lowercase();
    for (phrase, replacement) in PHRASE_REWRITES {
        result = result.replace(phrase, replacement);
    }
    result
}

/// A row consisting purely of weekday names — these are calendar header rows
/// that should not be returned as answers.
fn is_weekday_row(paragraph: &str) -> bool {
    const DAYS: &[&str] = &["SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"];
    let upper = paragraph.to_uppercase();
    // If every token in the row is a weekday name or a pipe separator, it's a header row.
    // An empty alphabetic-token list (e.g. pure numbers like "12") must NOT be treated as
    // a weekday row — `.all()` on an empty iterator returns `true`, so we guard explicitly.
    let alpha_tokens: Vec<&str> = upper
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| !t.is_empty())
        .collect();
    !alpha_tokens.is_empty() && alpha_tokens.iter().all(|t| DAYS.contains(t))
}
fn synonym_expand(word: &str) -> Vec<&'static str> {
    for (key, group) in SYNONYMS {
        if *key == word || group.contains(&word) {
            return group.to_vec();
        }
    }
    vec![]
}

/// Words that indicate a query is looking for a public event/date (not a meeting).
const EVENT_WORDS: &[&str] = &[
    "ceremony", "held", "date", "when", "holiday", "day", "graduation",
    "celebration", "event", "start", "begin", "end", "close", "open",
    "recess", "semester", "term", "exam", "registration", "classes",
];

/// Score how relevant `paragraph` is to `question` using keyword overlap.
///
/// Returns the fraction of non-stop question words found in the paragraph
/// (case-insensitive).  Synonym expansion, exact-phrase and standalone-token
/// bonuses are applied.  A committee-meeting penalty is applied when the query
/// is asking about an event but the paragraph is a committee meeting entry.
/// Range: [0.0, ∞) — base [0,1]; bonuses can push above 1; penalty can reduce.
fn keyword_relevance(question: &str, paragraph: &str) -> f32 {
    // Rewrite compound academic phrases ("second semester" → tokens that map
    // unambiguously to the correct term) before splitting into query words.
    let normalized = normalize_question(question);
    let q_words: Vec<String> = normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w: &&str| !w.is_empty() && !STOP_WORDS.contains(w))
        .map(str::to_string)
        .collect();

    if q_words.is_empty() {
        return 0.0;
    }

    let para_lower = paragraph.to_lowercase();

    let hits = q_words.iter().filter(|w| {
        // Direct match.
        if para_lower.contains(w.as_str()) {
            return true;
        }
        // Synonym expansion match.
        synonym_expand(w).iter().any(|s| para_lower.contains(*s))
    }).count();

    let mut score = hits as f32 / q_words.len() as f32;

    // Exact-phrase bonus.
    let q_clean: String = question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != ' ')
        .collect::<Vec<_>>()
        .join(" ");
    if !q_clean.is_empty() && para_lower.contains(&q_clean) {
        score += 0.5;
    }

    // Split paragraph into standalone uppercase tokens once (reused below).
    let para_upper = paragraph.to_uppercase();
    let upper_toks: Vec<&str> = para_upper
        .split(|c: char| !c.is_alphabetic())
        .filter(|t| !t.is_empty())
        .collect();

    // Standalone-word bonus: prefer paragraphs where query words appear as
    // ALL-CAPS standalone tokens (public events) over committee name matches.
    let upper_hits = q_words.iter().filter(|w| {
        let wu = w.to_uppercase();
        upper_toks.iter().any(|tok| *tok == wu)
    }).count();
    score += (upper_hits as f32 / q_words.len() as f32) * 0.4;

    // Adjacent-pair bonus: if two consecutive query words appear as adjacent
    // ALL-CAPS tokens (e.g. ["SUMMER","GRADUATION"]), this is a strong signal
    // the paragraph is an event headline rather than a committee name.
    //
    // We check every pair of query words, not just consecutive query order,
    // since the user might write "graduation ceremony" or "end of year graduation".
    let q_upper: Vec<String> = q_words.iter().map(|w| w.to_uppercase()).collect();
    let pair_bonus: f32 = q_upper.windows(2)
        .filter(|pair| {
            upper_toks.windows(2).any(|tok_pair| {
                tok_pair[0] == pair[0].as_str() && tok_pair[1] == pair[1].as_str()
            })
        })
        .count() as f32 * 0.5;
    score += pair_bonus;

    // Committee-meeting penalty: when the query is about an event/date and the
    // paragraph is a committee/planning entry, reduce score by 0.4.
    // This prevents "Graduation Planning Committee" rows from outranking
    // "SUMMER GRADUATION" event rows.
    let query_wants_event = q_words.iter().any(|w| EVENT_WORDS.contains(&w.as_str()));
    if query_wants_event && para_lower.contains("committee") {
        score -= 0.4;
    }

    score.max(0.0)
}

/// Extract the month-year prefix from an enriched paragraph like
/// `"December 2026: ..."` → `Some("December 2026")`.
fn month_prefix(paragraph: &str) -> Option<&str> {
    // Prefix format is "MonthName YYYY: rest" or "MonthName: rest"
    if let Some(colon_pos) = paragraph.find(": ") {
        let prefix = &paragraph[..colon_pos];
        // Validate it actually starts with a month name.
        let upper = prefix.to_uppercase();
        let is_month = crate::data::loader::MONTH_NAMES
            .iter()
            .any(|m| upper.starts_with(m));
        if is_month {
            return Some(prefix);
        }
    }
    None
}

/// Strip the month-year prefix from an enriched paragraph, returning the
/// raw calendar content.
fn strip_month_prefix(paragraph: &str) -> &str {
    if let Some(colon_pos) = paragraph.find(": ") {
        let prefix = &paragraph[..colon_pos];
        let upper = prefix.to_uppercase();
        let is_month = crate::data::loader::MONTH_NAMES
            .iter()
            .any(|m| upper.starts_with(m));
        if is_month {
            return paragraph[colon_pos + 2..].trim();
        }
    }
    paragraph.trim()
}

/// Extract a 4-digit year (1900-2199) from a question string, if present.
fn extract_year_from_query(question: &str) -> Option<u32> {
    let mut buf = String::new();
    for c in question.chars() {
        if c.is_ascii_digit() {
            buf.push(c);
            if buf.len() == 4 {
                if let Ok(y) = buf.parse::<u32>() {
                    if (1900..=2199).contains(&y) {
                        return Some(y);
                    }
                }
                buf.clear();
            }
        } else {
            buf.clear();
        }
    }
    None
}

/// Parse the leading day number or day-range from a calendar cell string.
///
/// Examples:
///   `"27 FREEDOM DAY"`    → `Some(("27",    "FREEDOM DAY"))`
///   `"13-18 AUTUMN GRAD"` → `Some(("13-18", "AUTUMN GRAD"))`
///
/// Returns `None` when the cell doesn't start with a day number.
fn split_day_from_cell(cell: &str) -> Option<(String, &str)> {
    let trimmed = cell.trim();
    // Collect leading digits (1-2 chars).
    let end = trimmed.find(|c: char| !c.is_ascii_digit()).unwrap_or(trimmed.len());
    if end == 0 || end > 2 {
        return None;
    }
    let day: u32 = trimmed[..end].parse().ok()?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let after = &trimmed[end..];
    // Check for range notation: "DD-DD rest"
    if after.starts_with('-') {
        let range_part = &after[1..];
        let end2 = range_part
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(range_part.len());
        if end2 > 0 && end2 <= 2 {
            if let Ok(day2) = range_part[..end2].parse::<u32>() {
                if (1..=31).contains(&day2) {
                    let rest = range_part[end2..].trim();
                    return Some((format!("{}-{}", day, day2), rest));
                }
            }
        }
    }
    let rest = after.trim();
    Some((day.to_string(), rest))
}

/// Convert a month name (any case) to a 1-based month number, or `None`.
fn month_name_to_number(name: &str) -> Option<u32> {
    match name.to_uppercase().as_str() {
        "JANUARY"   => Some(1),  "FEBRUARY" => Some(2),  "MARCH"    => Some(3),
        "APRIL"     => Some(4),  "MAY"      => Some(5),  "JUNE"     => Some(6),
        "JULY"      => Some(7),  "AUGUST"   => Some(8),  "SEPTEMBER"=> Some(9),
        "OCTOBER"   => Some(10), "NOVEMBER" => Some(11), "DECEMBER" => Some(12),
        _ => None,
    }
}

/// Return the weekday name (e.g. `"Monday"`) for a given date.
///
/// Uses the Tomohiko Sakamoto algorithm — no external dependency needed.
/// Returns `None` for invalid dates.
fn weekday_name(year: u32, month: u32, day: u32) -> Option<&'static str> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let (y, m) = if month < 3 { (year - 1, month) } else { (year, month) };
    let dow = (y + y/4 - y/100 + y/400 + T[(m - 1) as usize] + day) % 7;
    // 0=Sun, 1=Mon, …, 6=Sat
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    Some(DAYS[dow as usize])
}

/// Parse a "Month YYYY" prefix into (month_number, year).
///
/// E.g. `"June 2024"` → `Some((6, 2024))`.
fn parse_month_year(month_year: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = month_year.split_whitespace().collect();
    let month = month_name_to_number(parts.first().copied()?)?;
    let year  = parts.get(1).and_then(|s| s.parse::<u32>().ok())?;
    Some((month, year))
}

/// Convert an internal `"Month YYYY: DD EVENT TEXT"` string to the
/// human-friendly format `"DD Month YYYY - EVENT TEXT"`.
///
/// For single-day events the weekday name is prepended:
///   `"Sunday 16 June 2024 - YOUTH DAY"`
///
/// For date-range events the weekday is intentionally omitted:
///   `"13-18 April 2026 - AUTUMN GRADUATION"`
///
/// Supports day ranges: `"13-18 AUTUMN GRADUATION"` → `"13-18 April 2026 - AUTUMN GRADUATION"`.
/// If the cell has no leading day number, falls back to `"Month YYYY - TEXT"`.
fn pretty_date(month_year: &str, cell_body: &str) -> String {
    // Deduplicate repeated event names (calendar cells often repeat the label
    // in two columns, e.g. "SUMMER GRADUATION SUMMER GRADUATION").
    let dedup: String = {
        let words: Vec<&str> = cell_body.split_whitespace().collect();
        let half = words.len() / 2;
        let deduped =
            if half >= 2 && words[..half] == words[half..] { &words[..half] } else { &words[..] };
        deduped.join(" ")
    };

    // split_day_from_cell now returns a String day key (e.g. "13" or "13-18").
    match split_day_from_cell(&dedup) {
        Some((day_str, event)) if !event.is_empty() => {
            // Only prepend the weekday for single-day entries (no "-" range).
            let prefix = if !day_str.contains('-') {
                if let (Ok(d), Some((mo, yr))) = (day_str.parse::<u32>(), parse_month_year(month_year)) {
                    weekday_name(yr, mo, d).map(|w| format!("{w} ")).unwrap_or_default()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            format!("{prefix}{day_str} {month_year} - {event}")
        }
        Some((day_str, _)) => {
            format!("{day_str} {month_year}")
        }
        None => {
            format!("{month_year} - {dedup}")
        }
    }
}

/// Pick the most relevant cell from a `" | "`-separated table row and return
/// a formatted date string.
///
/// For spanning event bars (e.g. "AUTUMN GRADUATION" anchored at day 15 but
/// covering Mon Apr 13 – Sat Apr 18), we expand the date range by scanning
/// left over consecutive bare-day cells and extending the end to the last day
/// in the week row.
///
/// Example input:  "December 2026: 9 SUMMER GRADUATION | 10 End of 16 days"
/// Example output: "9 December 2026 - SUMMER GRADUATION"
fn best_cell(question: &str, paragraph: &str) -> String {
    let prefix = month_prefix(paragraph).unwrap_or("").to_string();
    let content = strip_month_prefix(paragraph);

    // Split on table cell separators, filtering out pure weekday-name cells.
    let cells: Vec<&str> = content
        .split(" | ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter(|s| !is_weekday_row(s))
        .collect();

    if cells.is_empty() {
        return paragraph.to_string();
    }

    let best = cells
        .iter()
        .max_by(|a, b| {
            keyword_relevance(question, a)
                .partial_cmp(&keyword_relevance(question, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(cells[0]);

    // ── Date-range expansion for spanning event bars ──────────────────────────
    // When the best cell is an event label anchored at a specific day (e.g.,
    // "15 AUTUMN GRADUATION"), the visual bar may span several more days in
    // the same week row.  Expand the range by:
    //   1. Collecting all day-numbers in the row.
    //   2. If the row is a complete 7-day week (max-min == 6) skip the first
    //      day (Sunday) when computing the weekday start.
    //   3. Scanning left from the anchor over consecutive bare-day cells
    //      (cells that contain only a day number with no event text).  Stop
    //      as soon as a non-bare cell is encountered.
    //   4. Using the last day in the row as the range end.
    let best_for_pretty: String = if let Some((day_str, event)) = split_day_from_cell(best) {
        // Only expand when the best cell contains an event label (not just a day
        // number) and the day portion is a plain single number (no range yet).
        let already_ranged = day_str.contains('-');
        if !event.is_empty() && !already_ranged {
            if let Ok(anchor_day) = day_str.parse::<u32>() {
                // All day-numbers in this week row.
                let row_days: Vec<u32> = cells
                    .iter()
                    .filter_map(|c| {
                        let end = c
                            .find(|ch: char| !ch.is_ascii_digit())
                            .unwrap_or(c.len());
                        if end == 0 || end > 2 {
                            return None;
                        }
                        c[..end].parse::<u32>().ok().filter(|d| (1..=31).contains(d))
                    })
                    .collect();

                if row_days.len() >= 2 {
                    let min_day = *row_days.iter().min().unwrap();
                    let max_day = *row_days.iter().max().unwrap();
                    // First weekday: skip Sunday when the row covers a full week.
                    let first_weekday =
                        if max_day.saturating_sub(min_day) == 6 { min_day + 1 } else { min_day };

                    // Scan left from anchor over bare cells (digit-only cells).
                    let mut range_start = anchor_day;
                    let mut check = anchor_day;
                    while check > first_weekday {
                        let prev = check - 1;
                        // A cell is "bare" when its trimmed text is just the day number.
                        let is_bare = cells
                            .iter()
                            .any(|c| c.trim() == prev.to_string().as_str());
                        if is_bare {
                            range_start = prev;
                            check = prev;
                        } else {
                            break; // non-bare cell encountered — stop expansion
                        }
                    }

                    // Only produce a range when leftward expansion actually occurred
                    // (i.e. the bar genuinely spans multiple days starting before
                    // the anchor cell).  Single-day events like FREEDOM DAY must
                    // not be extended to the end of the week row.
                    if range_start < anchor_day {
                        format!("{}-{} {}", range_start, max_day, event)
                    } else {
                        // No leftward expansion — single-day event.
                        format!("{} {}", anchor_day, event)
                    }
                } else {
                    best.to_string()
                }
            } else {
                best.to_string()
            }
        } else {
            best.to_string()
        }
    } else {
        best.to_string()
    };

    if prefix.is_empty() {
        best_for_pretty
    } else {
        pretty_date(&prefix, &best_for_pretty)
    }
}

/// Format a clean human-readable answer from a raw extracted span and its
/// source paragraph (which carries the month prefix).
fn format_answer(raw: &str, source_paragraph: &str) -> String {
    let prefix = month_prefix(source_paragraph).map(str::to_string);

    match prefix {
        Some(p) => pretty_date(&p, raw),
        None    => raw.split_whitespace().collect::<Vec<_>>().join(" "),
    }
}

/// Convert a raw filename like `calader_2026.docx` into a human-friendly label
/// such as `"calader 2026 document"`.
fn friendly_doc_name(filename: &str) -> String {
    let stem = filename
        .trim_end_matches(".docx")
        .trim_end_matches(".doc")
        .replace('_', " ")
        .replace('-', " ");
    format!("{stem} document")
}

/// Extract the (min_day, max_day) present in a table row string such as
/// `"12 | 13 | 14 | 15 | 16 Sisonke Supervision | 17 | 18"` → `Some((12, 18))`.
/// Used to annotate event-label rows with their spanning date range.
fn extract_day_range(row: &str) -> Option<(u32, u32)> {
    let mut days: Vec<u32> = Vec::new();
    for cell in row.split(" | ") {
        let trimmed = cell.trim();
        let end = trimmed
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(trimmed.len());
        if end > 0 && end <= 2 {
            if let Ok(d) = trimmed[..end].parse::<u32>() {
                if (1..=31).contains(&d) {
                    days.push(d);
                }
            }
        }
    }
    if days.len() >= 2 {
        Some((*days.iter().min()?, *days.iter().max()?))
    } else {
        None
    }
}

/// Attach a month-year prefix to every content row in a document's paragraph list.
///
/// Calendar documents contain standalone header rows like "DECEMBER 2026" followed
/// by table content rows with no date context.  This function detects those headers
/// and prepends them to subsequent rows so that retrieval can match year/month.
///
/// Example:
///   "DECEMBER 2026"  → kept as-is (will be filtered by min-length later)
///   "9 SUMMER GRADUATION | 10 End of 16 days"  → "December 2026: 9 SUMMER GRADUATION | 10 End"
fn enrich_doc_paragraphs(
    paragraphs: &[String],
    filename: &str,
    year: Option<u32>,
) -> Vec<(String, String, Option<u32>)> {
    let mut result = Vec::with_capacity(paragraphs.len());
    let mut current_month: Option<String> = None;

    for para in paragraphs {
        let trimmed = para.trim();
        // A month header is a row whose content is just "MONTHNAME" or "MONTHNAME YYYY".
        let upper = trimmed.to_uppercase();
        let token_count = trimmed.split_whitespace().count();
        let is_month_header = token_count <= 3
            && crate::data::loader::MONTH_NAMES.iter().any(|m| upper.starts_with(m));

        if is_month_header {
            // Convert to title-case so the prefix matches what `month_prefix` expects.
            let title: String = trimmed
                .split_whitespace()
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str().to_lowercase().as_str(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            current_month = Some(title);
            result.push((para.to_string(), filename.to_string(), year));
        } else if let Some(ref month) = current_month {
            // Prefix content rows with `Month YYYY: ` context.
            let enriched = format!("{}: {}", month, trimmed);
            result.push((enriched, filename.to_string(), year));
        } else {
            result.push((para.to_string(), filename.to_string(), year));
        }
    }
    // ── Post-process: annotate pure event-label rows with their date range ──
    //
    // After fixing the drawing text extractor, the docx table rows now include
    // rows like "April 2026: AUTUMN GRADUATION" (drawn text box content, no
    // day numbers) alongside the normal day-number rows such as
    // "April 2026: 12 | 13 | 14 | 15 | 16 Sisonke | 17 | 18".
    //
    // We look ahead to the next row that is in the same month AND contains a
    // day range, then annotate the event-label row with that range so the
    // model can answer questions like "When is autumn graduation?" with
    // "13-18 April 2026 - AUTUMN GRADUATION" instead of a bare label.
    let len = result.len();
    for i in 0..len {
        let content = {
            let (para, _, _) = &result[i];
            strip_month_prefix(para.as_str()).to_string()
        };

        // Only process rows that are a pure event label:
        //   • no " | " cell separator (single-cell, not a weekly grid row)
        //   • no leading day number
        //   • content is non-trivial
        //   • all alphabetic characters are uppercase (event heading)
        let is_pure_label = !content.contains(" | ")
            && split_day_from_cell(&content).is_none()
            && content.len() > 3
            && content
                .chars()
                .filter(|c| c.is_alphabetic())
                .all(|c| c.is_uppercase());

        if !is_pure_label {
            continue;
        }

        // Extract the current month prefix once for comparison.
        let month_pre = {
            let (para, _, _) = &result[i];
            month_prefix(para.as_str()).map(str::to_string)
        };

        // Scan surrounding rows (same month) in both directions for the
        // week-grid row.  The floating label paragraph may occur either
        // before or after its host weekly row in the docx XML order.
        let mut found_range: Option<(u32, u32)> = None;
        // Look forward up to 8 rows.
        for j in (i + 1)..(i + 9).min(len) {
            let (next_para, _, _) = &result[j];
            let same_month = match (&month_pre, month_prefix(next_para.as_str())) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            if !same_month {
                break;
            }
            let next_content = strip_month_prefix(next_para.as_str());
            if let Some(range) = extract_day_range(next_content) {
                found_range = Some(range);
                break;
            }
        }
        // Also look backwards up to 8 rows.
        if found_range.is_none() {
            let back_start = if i >= 8 { i - 8 } else { 0 };
            for j in (back_start..i).rev() {
                let (prev_para, _, _) = &result[j];
                let same_month = match (&month_pre, month_prefix(prev_para.as_str())) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };
                if !same_month {
                    break;
                }
                let prev_content = strip_month_prefix(prev_para.as_str());
                if let Some(range) = extract_day_range(prev_content) {
                    found_range = Some(range);
                    break;
                }
            }
        }

        if let Some((min_d, max_d)) = found_range {
            let (para, _, _) = &mut result[i];
            let prefix_str = month_prefix(para.as_str()).map(str::to_string);
            let new_content = format!("{}-{} {}", min_d, max_d, content);
            para.clear();
            match prefix_str {
                Some(p) => para.push_str(&format!("{}: {}", p, new_content)),
                None => para.push_str(&new_content),
            }
        }
    }

    result
}

// ── Interactive CLI loop ──────────────────────────────────────────────────────

/// Load model from `model_dir`, extract context from .docx files in `docs_dir`,
/// then enter an interactive Q&A loop.
pub fn run_inference(docs_dir: &Path, model_dir: &Path) {
    println!("[infer] Loading model from '{}'…", model_dir.display());
    let (model, vocab) = load_model(model_dir);
    println!("[infer] Model loaded.\n");

    println!("[infer] Scanning documents in '{}'…", docs_dir.display());
    let docs = load_all_docx(docs_dir);
    if docs.is_empty() {
        eprintln!("[infer] No .docx files found. Aborting.");
        return;
    }

    // Collect every paragraph, keeping track of its source document and year.
    // Run through per-document month enrichment so that calendar content rows
    // carry a "Month YYYY: " prefix enabling date-aware retrieval.
    // Also drop pure weekday-name rows (calendar column headers).
    let paragraphs: Vec<(String, String, Option<u32>)> = docs
        .iter()
        .flat_map(|d| enrich_doc_paragraphs(&d.paragraphs, &d.filename, d.year))
        .filter(|(p, _, _)| {
            let content = strip_month_prefix(p.as_str());
            p.trim().len() > 10 && !is_weekday_row(content)
        })
        .collect();

    println!(
        "[infer] Loaded {} paragraph(s) from {} document(s).\n",
        paragraphs.len(),
        docs.len()
    );

    // Interactive loop.
    loop {
        print!("Question (or 'exit'): ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        let n = io::stdin().read_line(&mut line).unwrap_or(0);
        if n == 0 {
            break; // EOF
        }
        let question = line.trim();

        if question.eq_ignore_ascii_case("exit") || question.eq_ignore_ascii_case("quit") {
            break;
        }
        if question.is_empty() {
            continue;
        }

        // ── Step 1: narrow to the relevant year's document if mentioned ──
        let query_year = extract_year_from_query(question);
        let search_pool: Vec<&(String, String, Option<u32>)> = paragraphs
            .iter()
            .filter(|(_, _, doc_yr)| {
                // Keep paragraph if either the query has no year, or the
                // document year matches the queried year.
                match (query_year, doc_yr) {
                    (Some(qy), Some(dy)) => qy == *dy,
                    _ => true,
                }
            })
            .collect();

        // ── Step 2: rank filtered paragraphs by keyword relevance ─────────
        // Exclude pure weekday-header rows from participation.
        let mut scored: Vec<(f32, &(String, String, Option<u32>))> = search_pool
            .iter()
            .filter(|(p, _, _)| !is_weekday_row(p))
            .map(|triple| (keyword_relevance(question, &triple.0), *triple))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // ── Debug: show top-3 scored paragraphs (first 120 chars each) ────
        println!("\x1b[2m[debug] top candidates:");
        for (rank, (score, triple)) in scored.iter().take(3).enumerate() {
            let preview: String = triple.0.chars().take(120).collect();
            println!("  {}. score={:.3}  \"{}\"", rank + 1, score, preview);
        }
        println!("[/debug]\x1b[0m");

        // ── Step 3: run the model on the top-5 most relevant paragraphs ───
        let top_k: Vec<(f32, &(String, String, Option<u32>))> = scored.into_iter().take(5).collect();

        let best_model = top_k
            .iter()
            .filter_map(|(kw_score, triple)| {
                answer_question(&model, &vocab, question, &triple.0, *kw_score)
                    .map(|(ans, score)| (ans, score, triple.0.clone(), triple.1.clone()))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best_model {
            Some((answer, score, source_para, source_doc)) => {
                // keyword relevance is in [0, ~2.5]; map to 30-95% display range.
                let pct        = ((score * 30.0 + 30.0) as u32).clamp(5, 95);
                let doc_label  = friendly_doc_name(&source_doc);
                let clean      = format_answer(&answer, &source_para);
                println!("\n\x1b[32;1mAnswer\x1b[0m [{pct}% conf]: Based on the {doc_label}, {clean}\n");
            }
            None => {
                // ── Step 4: cell-level keyword fallback ────────────────────
                if let Some((relevance, triple)) = top_k.first() {
                    if *relevance > 0.0 {
                        let cell       = best_cell(question, &triple.0);
                        let doc_label  = friendly_doc_name(&triple.1);
                        println!("\n\x1b[33mAnswer\x1b[0m [retrieval]: Based on the {doc_label}, {cell}\n");
                    } else {
                        println!("\n\x1b[33mAnswer\x1b[0m [0% conf]: (not found in document)\n");
                    }
                } else {
                    println!("\n\x1b[33mAnswer\x1b[0m [0% conf]: (not found in document)\n");
                }
            }
        }
    }

    println!("[infer] Goodbye.");
}
