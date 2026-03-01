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
    // NOTE: "many" is intentionally NOT listed here — it is a discriminative
    // word in "how many" count queries and must survive the stop-word filter.
    "much", "some", "any", "all", "their", "there", "here",
    "about", "into", "than", "then", "also", "each", "per",
];

/// Synonym/phrase groups: any question word that matches a group key will also
/// be credited when the paragraph contains any word in that group's value list.
const SYNONYMS: &[(&str, &[&str])] = &[
    ("graduation",   &["graduation", "graduate", "graduates", "ceremony", "commencement", "summer graduation", "autumn graduation"]),
    ("ceremony",     &["ceremony", "graduation", "commencement"]),
    ("summer",       &["summer", "december", "summer graduation"]),
    // "year" on its own should NOT pull in December — that caused "When does the
    // academic year begin?" to rank December paragraphs above January ones.
    // End-of-year graduation is handled by the "graduation"/"summer" synonyms.
    ("year",         &["year", "annual"]),
    ("end",          &["end", "summer", "december", "final"]),
    ("autumn",       &["autumn", "fall", "april", "autumn graduation"]),
    ("workers",      &["workers", "labour", "labor"]),
    ("labour",       &["labour", "workers", "labor"]),
    ("holiday",      &["holiday", "holidays", "recess", "public", "national", "day"]),
    ("holidays",     &["holidays", "holiday", "recess", "public", "national", "day"]),
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
    ("begin",        &["begin", "start", "start of term", "start of year", "first day"]),
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
    ("public",       &["public", "holiday", "holidays", "day"]),
    ("june",         &["june", "winter", "youth day"]),
    // ── Count / frequency queries ────────────────────────────────────────────
    // "how many times" / "how often" should match meeting/session paragraphs.
    ("many",         &["many", "times", "number", "count", "total", "often"]),
    ("times",        &["times", "many", "often", "frequency", "occurrences", "sessions", "meetings"]),
    ("often",        &["often", "times", "many", "frequently", "regularly"]),
    ("count",        &["count", "number", "total", "many"]),
    ("meeting",      &["meeting", "meetings", "session", "sessions", "board", "committee", "forum"]),
    ("meetings",     &["meetings", "meeting", "sessions", "board", "committee", "forum"]),
    ("board",        &["board", "committee", "governance", "management", "council", "forum"]),
    ("governance",   &["governance", "board"]),
    // Handles common document typo: "SARETEC" instead of "SARTEC".
    ("sartec",       &["sartec", "saretec"]),
    ("saretec",      &["sartec", "saretec"]),
    // ── Committee acronyms ──────────────────────────────────────────────────
    ("hdc",          &["hdc", "higher degrees committee", "higher degrees"]),
    ("higher",       &["higher", "hdc", "higher degrees"]),
    // ── Duration queries ─────────────────────────────────────────────────────
    // "how long is the first semester" → look for term-start/end paragraphs.
    ("long",         &["long", "duration", "length", "weeks", "days", "months", "period"]),
    ("duration",     &["duration", "long", "length", "period", "weeks", "days"]),
    ("length",       &["length", "duration", "long", "weeks", "period"]),
    ("weeks",        &["weeks", "week", "days", "duration", "long", "period"]),
    ("days",         &["days", "day", "weeks", "duration", "period"]),
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
    // ── Committee acronym expansions ─────────────────────────────────────
    // Expand before keyword scoring so entity extraction and matching both
    // see the full committee name that appears in the calendar documents.
    ("hdc",                "higher degrees committee"),
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

    // Committee-meeting penalty: when the query is about an event/date (but is
    // NOT a count/frequency/meeting query itself) and the paragraph is a
    // committee/planning entry, reduce score by 0.4.
    // This prevents "Graduation Planning Committee" rows from outranking
    // "SUMMER GRADUATION" event rows.
    //
    // EXCEPTION: "how many times", "how often", "meeting", "board" queries are
    // explicitly *seeking* committee-meeting paragraphs, so the penalty must
    // not fire for them.
    const COUNT_WORDS: &[&str] = &[
        "many", "times", "often", "count", "number", "meetings", "meeting",
        "board", "sessions", "session", "governance", "forum",
    ];
    let query_wants_event = q_words.iter().any(|w| EVENT_WORDS.contains(&w.as_str()));
    let query_wants_count = q_words.iter().any(|w| COUNT_WORDS.contains(&w.as_str()));
    if query_wants_event && !query_wants_count && para_lower.contains("committee") {
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
                    // the anchor cell).  Single-day events like FREEDOM DAY or
                    // CHRISTMAS DAY must not be extended to the end of the week row.
                    //
                    // Guard: if the event text ends with the word "DAY" it is a
                    // single-day public holiday — never expand regardless of how
                    // many bare cells precede it in the week row.
                    let last_word = event.split_whitespace().next_back().unwrap_or("");
                    let is_single_day_holiday = last_word.eq_ignore_ascii_case("day");
                    if range_start < anchor_day && !is_single_day_holiday {
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

// ── Matched-event segment extraction ─────────────────────────────────────────

/// Given an entity query (e.g. `"SARTEC Governance"`) and a raw calendar cell
/// body (e.g. `"International Women's Day SARETEC Governance Board Meeting (09:00)"`),
/// return only the sub-string that is the matched event, including its time
/// indicator if present — and nothing before or after it.
///
/// Examples:
///   entity = "SARTEC Governance"
///   cell   = "International Women's Day SARETEC Governance Board Meeting (09 :00)"
///   result = "SARETEC Governance Board Meeting (09 :00)"
///
///   entity = "SARTEC Governance"
///   cell   = "SARETEC Governance Board Meeting (09:00) Waste & Recycling Awareness Campaign"
///   result = "SARETEC Governance Board Meeting (09:00)"
///
///   entity = "Senate"
///   cell   = "Senate (12:00) Waste & Recycling Awareness Campaign"
///   result = "Senate (12:00)"
fn extract_matched_event(entity_query: &str, cell_body: &str) -> String {
    if entity_query.trim().is_empty() {
        return cell_body.trim().to_string();
    }

    let cell_lower = cell_body.to_lowercase();
    let entity_words: Vec<&str> = entity_query.split_whitespace().collect();

    // ── Strategy: find the position in the cell where the entity phrase starts.
    //
    // We try to match the entity as a "phrase" (all words appearing together in
    // order) to avoid anchoring on an unrelated earlier occurrence of a shared
    // word like "governance" appearing in "Council Governance and Ethics
    // Committee" before "SARETEC Governance Board Meeting".
    //
    // Step 1: Build candidate surface forms for each entity word.
    //   For spelling-variant synonyms (sartec/saretec) we include them.
    //   We do NOT include broad semantic synonyms (governance→board) here
    //   because those would anchor on the wrong event.
    let word_candidates: Vec<Vec<String>> = entity_words.iter().map(|w| {
        let wl = w.to_lowercase();
        // Only include the word itself plus synonyms that are recognisably
        // the same surface form (length within ±2 chars).
        let mut cands = vec![wl.clone()];
        for syn in synonym_expand(&wl) {
            let sl = syn.to_lowercase();
            let len_diff = (sl.len() as isize - wl.len() as isize).unsigned_abs();
            if len_diff <= 2 {
                cands.push(sl);
            }
        }
        cands
    }).collect();

    // Step 2: Scan cell_lower byte-by-byte to find a span where EACH entity
    // word (in order) appears, allowing arbitrary text between consecutive
    // words (they just must appear in left-to-right order with no backtracking).
    // We report the start position of the FIRST word match.
    let start_pos: Option<usize> = {
        let mut best: Option<usize> = None;
        // Try every occurrence of the first entity word as a possible anchor.
        for first_cand in &word_candidates[0] {
            let mut search_from = 0;
            while let Some(rel) = cell_lower[search_from..].find(first_cand.as_str()) {
                let anchor = search_from + rel;
                // Try to match remaining words in order after anchor.
                let mut pos = anchor + first_cand.len();
                let mut all_match = true;
                for cands in &word_candidates[1..] {
                    // Find the earliest occurrence of any candidate for this word.
                    let found = cands.iter().filter_map(|c| {
                        cell_lower[pos..].find(c.as_str()).map(|r| (r, c.len()))
                    }).min_by_key(|(r, _)| *r);
                    match found {
                        Some((r, clen)) => pos += r + clen,
                        None => { all_match = false; break; }
                    }
                }
                if all_match {
                    best = Some(match best {
                        None    => anchor,
                        Some(p) => p.min(anchor),
                    });
                    break; // first (leftmost) occurrence of this candidate is enough
                }
                search_from = anchor + 1;
            }
        }
        best
    };

    // Byte offset where the matched event starts (fallback: beginning of cell).
    let start = {
        let s = start_pos.unwrap_or(0);
        // Ensure valid UTF-8 boundary.
        let mut s2 = s;
        while s2 > 0 && !cell_body.is_char_boundary(s2) { s2 -= 1; }
        s2
    };

    let from_start = &cell_body[start..];

    // Find the end: include up to and including the closing paren of the
    // first time indicator `(…:…)` or `(@…:…)`.
    let end = {
        let len = from_start.len();
        let mut result = len;
        let mut i = 0;
        let chars: Vec<char> = from_start.chars().collect();
        let char_count = chars.len();
        while i < char_count {
            if chars[i] == '(' {
                // Collect inner text up to matching ')'.
                let open_byte = from_start
                    .char_indices()
                    .nth(i)
                    .map(|(b, _)| b)
                    .unwrap_or(len);
                if let Some(close_rel) = from_start[open_byte..].find(')') {
                    let inner = &from_start[open_byte + 1..open_byte + close_rel];
                    let inner_t = inner.trim().trim_start_matches('@');
                    let is_time = inner_t.contains(':')
                        && inner_t.chars().any(|c| c.is_ascii_digit());
                    if is_time {
                        result = open_byte + close_rel + 1;
                        break;
                    }
                }
            }
            i += 1;
        }
        result
    };

    from_start[..end].trim().to_string()
}

// ── Count / frequency aggregation ────────────────────────────────────────────

/// Return `true` when the question is asking for a count or frequency rather
/// than a specific date span.
///
/// Matches patterns like:
///   "How many times does X meet?"
///   "How many board meetings does Y have?"
///   "How often does Z occur?"
/// Returns `true` when the question asks how many times the entity *itself*
/// meets ("how many times does X meet"), as opposed to a typed query like
/// "how many board meetings does X have".
fn is_self_meeting_query(question: &str) -> bool {
    let lower = question.to_lowercase();
    let wants_count = lower.contains("how many times") || lower.contains("how often");
    let has_type = lower.contains("board meeting")
        || lower.contains("committee meeting")
        || lower.contains("governance board");
    wants_count && !has_type
}

/// Extract a meeting-type qualifier that the user explicitly named.
///
/// "how many board meetings does SARTEC Governance have?" → `Some("board")`
/// "how many governance board meetings?"                  → `Some("governance board")`
/// "how many times does the Senate meet?"                 → `None`
fn meeting_type_qualifier(question: &str) -> Option<String> {
    let lower = question.to_lowercase();
    if lower.contains("governance board") {
        return Some("governance board".into());
    }
    if lower.contains("board meeting") || lower.contains("board meetings") {
        return Some("board".into());
    }
    if lower.contains("committee meeting") || lower.contains("committee meetings") {
        return Some("committee".into());
    }
    None
}

fn is_count_query(question: &str) -> bool {
    let lower = question.to_lowercase();
    // Any question that contains "how many" is a count / aggregation query.
    // Examples:
    //   "how many board meetings does SARTEC Governance have?"
    //   "how many times does the Senate meet?"
    //   "how many public holidays are in April 2026?"
    if lower.contains("how many") || lower.contains("how often") {
        return true;
    }
    false
}

/// Strip question-structure words from a count query, leaving only the entity
/// / subject words that should be searched for in the documents.
///
/// E.g.:
///   "How many times does the Senate meet in 2026?"
///       → "senate"
///   "How many board meetings does SARTEC Governance have?"
///       → "sartec governance"   ("board" and "meetings" are type-descriptors, not identifiers)
///   "How often does the Supply Chain Management Committee meet?"
///       → "supply chain management committee"
fn count_query_entity(question: &str) -> String {
    // ── Strategy: detect query pattern and extract the entity ─────────────────
    //
    // Pattern A – "how many [TYPE] does/do/did [ENTITY] have/meet/hold?"
    //   → entity = words AFTER does/do/did, TYPE are the words before it.
    //   → "how many board meetings does SARTEC Governance have?"
    //       → entity = "SARTEC Governance"
    //
    // Pattern B – "how many [TYPE] are/is/were [in/there] [TIME PERIOD]?"
    //   → entity = TYPE (the thing being counted), TIME PERIOD is a constraint.
    //   → "how many public holidays are in April 2026?"
    //       → entity = "public holidays"
    //
    // Pattern C – "how many times does [ENTITY] meet/occur/happen?"
    //   → entity = ENTITY (falls under Pattern A).
    //
    // Falls back to stripping structural words when no pivot is found.
    let lower = question.to_lowercase();
    let words_lower: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != ' ')
        .flat_map(|chunk| chunk.split_whitespace())
        .collect();
    let orig_words: Vec<&str> = question
        .split(|c: char| !c.is_alphanumeric() && c != ' ')
        .flat_map(|chunk| chunk.split_whitespace())
        .collect();

    // Trailing predicate / action words (appear at the end, after the entity).
    const TRAILING: &[&str] = &[
        "have", "has", "hold", "held", "meet", "meets", "occur", "occurs",
        "happen", "happens", "take", "takes", "schedule", "schedules",
        "convene", "convenes", "run", "runs", "in", "per", "year", "annually",
    ];

    // Detect pivot: find "does/do/did" (Pattern A) or "are/is/were" (Pattern B).
    let does_pivot = words_lower.iter().position(|w| matches!(*w, "does" | "do" | "did"));
    let are_pivot  = words_lower.iter().position(|w| matches!(*w, "are" | "is" | "were"));
    // "does/do/did" takes priority (Pattern A is more precise).
    let pivot_kind: Option<(usize, bool)> = match (does_pivot, are_pivot) {
        (Some(d), _)    => Some((d, false)),  // false = Pattern A (entity after pivot)
        (None, Some(a)) => Some((a, true)),   // true  = Pattern B (entity before pivot)
        _               => None,
    };

    let entity_tokens: Vec<String> = if let Some((p, pattern_b)) = pivot_kind {
        if pattern_b {
            // Pattern B: "how many [TYPE] are/is [in/there] [TIME]?"
            // Entity = TYPE words between "many" and the pivot.
            let many_pos = words_lower.iter().position(|w| *w == "many").unwrap_or(0);
            orig_words[many_pos + 1..p]
                .iter()
                .copied()
                .filter(|w| {
                    let wl = w.to_lowercase();
                    !matches!(wl.as_str(), "the" | "a" | "an")
                        && !(w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
                })
                .map(str::to_string)
                .collect()
        } else {
        // Pattern A: entity is AFTER the pivot.
        // Words after the pivot preserve casing.
        let after: Vec<&str> = orig_words
            .iter()
            .skip(p + 1)
            .copied()
            .collect();
        // Strip leading determiners "the", "a", "an".
        let after = {
            let skip = after.iter()
                .take_while(|w| matches!(w.to_lowercase().as_str(), "the" | "a" | "an"))
                .count();
            &after[skip..]
        };
        // Take words until a trailing predicate is encountered.
        after.iter()
            .copied()
            .take_while(|w| !TRAILING.contains(&w.to_lowercase().as_str()))
            .filter(|w| {
                // Drop bare 4-digit years.
                !(w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
            })
            .map(str::to_string)
            .collect()
        }
    } else {
        // Fallback: strip every known structural word.
        const STRIP: &[&str] = &[
            "how", "many", "times", "often", "does", "do", "did", "the", "a", "an",
            "meet", "meets", "meeting", "meetings", "board", "boards",
            "have", "has", "hold", "held", "occur",
            "occurs", "occurrences", "sessions", "session", "convene", "convenes",
            "please", "tell", "much", "there", "are", "is",
            "what", "when", "where", "who", "which", "why",
            "in", "on", "at", "to", "for", "of", "and", "or", "by", "from",
            "its", "it", "this", "that",
        ];
        question
            .split(|c: char| !c.is_alphanumeric() && c != ' ')
            .flat_map(|chunk| chunk.split_whitespace())
            .filter(|w| {
                let wl = w.to_lowercase();
                if STRIP.contains(&wl.as_str()) { return false; }
                if w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()) { return false; }
                true
            })
            .map(str::to_string)
            .collect()
    };

    entity_tokens.join(" ")
}

/// For Pattern B count queries ("how many X are in MONTH YEAR?"), extract the
/// month number the user is asking about.
///
/// "how many public holidays are in April 2026?" → `Some(4)`
/// "how many board meetings does SARTEC have?"   → `None`
fn count_scope_month(question: &str) -> Option<u32> {
    let lower = question.to_lowercase();
    // Only applies when there is NO "does/do/did" pivot (Pattern B queries).
    let has_does = lower.split_whitespace().any(|w| matches!(w, "does" | "do" | "did"));
    if has_does {
        return None;
    }
    // Look for a month name anywhere in the question.
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    for w in &words {
        if let Some(m) = month_name_to_number(w) {
            return Some(m);
        }
    }
    None
}

/// Extract a type-qualifier word from a count query that should serve as a
/// **hard mandatory filter** on matched cells.
///
/// For "how many board meetings does SARTEC Governance have?" the type term
/// is "board" — matched cells must contain "board" so that generic entries
/// like "Council Governance and Ethics Committee" are excluded.
///
/// Returns `None` when no specific type qualifier is present (e.g. "how many
/// times does X meet?"), meaning no additional cell filter is applied.
fn count_type_qualifier(question: &str) -> Option<String> {
    // Generic count words that are NOT meaningful type qualifiers.
    const GENERIC: &[&str] = &["times", "time", "often", "many", "occurrences",
                                "instances", "sessions", "session"];

    let lower = question.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != ' ')
        .flat_map(|chunk| chunk.split_whitespace())
        .collect();

    let many_pos  = words.iter().position(|w| *w == "many")?;

    // Pattern B: "how many X are/is in Y?" → entity = X, no extra qualifier
    // needed (X itself IS the type we're counting).
    let has_does = words[many_pos..].iter().any(|w| matches!(*w, "does" | "do" | "did"));
    if !has_does {
        return None;
    }

    // Pattern A: "how many TYPE does ENTITY have?" → extract TYPE qualifier.
    let pivot_pos = words[many_pos..].iter()
        .position(|w| matches!(*w, "does" | "do" | "did"))
        .map(|p| many_pos + p)?;

    // Type words are between "many" and the pivot.
    let type_words: Vec<&str> = words[many_pos + 1..pivot_pos]
        .iter()
        .copied()
        .filter(|w| !GENERIC.contains(w))
        .collect();

    if type_words.is_empty() {
        return None;
    }

    // Return the most specific (first non-generic) type word as the qualifier.
    Some(type_words[0].to_string())
}

/// Scan `search_pool` for every calendar *cell* that mentions the entity
/// described by `entity_query`.
///
/// Uses `keyword_relevance` (with synonym expansion) to match both the
/// individual week-row paragraph AND the individual cell within that row,
/// so broad terms like "Senate" correctly enumerate each Senate meeting
/// while narrow terms like "SARTEC Governance" pick up the specific entry
/// despite minor spelling variations.
///
/// Returns `(total_count, Vec<pretty_date_string>)`.
fn count_occurrences(
    entity_query: &str,
    question: &str,
    search_pool: &[&(String, String, Option<u32>)],
    scope_month: Option<u32>,
) -> (usize, Vec<String>) {
    if entity_query.trim().is_empty() {
        return (0, vec![]);
    }

    // Threshold: para-level is used as a coarse pre-filter (synonym expansion
    // allowed).  Cell-level scoring requires ≥70 % of entity words matched
    // directly or via synonym — strict enough to avoid false positives from
    // broad synonym chains (e.g. "governance" → "board") while still catching
    // spelling variants like "SARETEC" (matches "sartec" via synonym).
    const PARA_THRESHOLD: f32 = 0.45;
    const CELL_THRESHOLD: f32 = 0.70;

    // ── Precision flags derived from the original question ────────────────
    // self_meeting: "how many times does X meet?"  → entity must be the
    //   primary subject of the cell (appear at the start, directly followed
    //   by a time indicator like "(12:00)"), so sub-committees of X are
    //   excluded.
    let self_meeting = is_self_meeting_query(question);
    // type_qualifier: the specific type word extracted from the question
    // (e.g. "board" from "how many board meetings does X have?").
    // Applied as a hard cell filter to exclude false-positive matches.
    let type_qualifier: Option<String> = count_type_qualifier(question);

    let mut dates: Vec<String> = Vec::new();

    for (para, _doc, _yr) in search_pool.iter().copied() {
        // Fast paragraph-level filter.
        if keyword_relevance(entity_query, para) < PARA_THRESHOLD {
            continue;
        }

        let prefix   = month_prefix(para.as_str()).map(str::to_string);
        let content  = strip_month_prefix(para.as_str());

        // Scope-month filter: if the query asked about a specific month
        // (e.g. "in April"), only count entries from that month.
        if let Some(sm) = scope_month {
            let para_month = prefix.as_deref()
                .and_then(parse_month_year)
                .map(|(m, _)| m);
            match para_month {
                Some(pm) if pm == sm => { /* in scope – continue */ }
                _ => continue,
            }
        }

        // Walk cells in the week row.
        let cells: Vec<&str> = content
            .split(" | ")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|s| !is_weekday_row(s))
            .collect();

        for cell in cells {
            // Skip bare day-number cells (no event text).
            if cell.trim().chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            if keyword_relevance(entity_query, cell) < CELL_THRESHOLD {
                continue;
            }

            // ── Self-meeting precision filter ────────────────────────────
            // "how many times does the Senate meet?"  →  the entity name must
            // appear at the *start* of the cell body (after any leading day
            // number) and be followed directly by a time indicator "(hh:mm)"
            // or "(@hh:mm)", with nothing else between.  This excludes cells
            // like "Senate Higher Degrees Committee (09:00)" whose primary
            // subject is the sub-committee, not the Senate itself.
            //
            // EXCEPTION: when the entity itself is a multi-word specific name
            // (≥ 3 words, e.g. "higher degrees committee"), the starts-with
            // guard is too strict — the entity may appear after a short prefix
            // like "Senate " in the cell body.  In that case keyword_relevance
            // at CELL_THRESHOLD already ensures precision; we skip the guard.
            if self_meeting && entity_query.split_whitespace().count() <= 2 {
                let cell_body: &str = match split_day_from_cell(cell) {
                    Some((_, rest)) => rest,
                    None            => cell,
                };
                let entity_lower   = entity_query.to_lowercase();
                let cell_body_lower = cell_body.to_lowercase();
                let direct = cell_body_lower.starts_with(&entity_lower) && {
                    let after = cell_body[entity_lower.len()..].trim();
                    after.is_empty() || after.starts_with('(') || after.starts_with('@')
                };
                if !direct {
                    continue;
                }
            }

            // ── Type qualifier hard filter ────────────────────────────────
            // "how many board meetings does SARTEC Governance have?" → only
            // keep cells that explicitly contain the type qualifier word
            // (e.g. "board").  This removes false positives like
            // "Council Governance and Ethics Committee" (no "board") while
            // correctly keeping "SARETEC Governance Board Meeting".
            // When no specific qualifier was found in the question (e.g. "how
            // many times does X meet?") no extra filter is applied.
            if let Some(ref q) = type_qualifier {
                if !cell.to_lowercase().contains(q.as_str()) {
                    continue;
                }
            }

            // Format a human-readable date, showing ONLY the matched event
            // segment (not the full cell which may include unrelated events).
            let cell_for_date: String = match split_day_from_cell(cell) {
                Some((day_str, body)) => {
                    let segment = extract_matched_event(entity_query, body);
                    format!("{} {}", day_str, segment)
                }
                None => extract_matched_event(entity_query, cell),
            };
            let formatted = match &prefix {
                Some(p) => pretty_date(p, &cell_for_date),
                None    => cell_for_date,
            };
            dates.push(formatted);
        }
    }

    let count = dates.len();
    (count, dates)
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

        // ── Step 1b: count / frequency shortcut ───────────────────────────
        // "How many times does X meet?" / "How often does Y occur?" →
        // aggregate matching cells across all paragraphs instead of
        // extracting a single span.
        if is_count_query(question) {
            // Expand acronyms / phrase rewrites before entity extraction so
            // that e.g. "HDC" → "Higher Degrees Committee" is resolved.
            // The original question is kept for behavioural flags (self_meeting,
            // type_qualifier, scope_month) to avoid changing query semantics.
            let norm_q = normalize_question(question);
            let entity = count_query_entity(&norm_q);
            let scope_month = count_scope_month(question);
            let (count, dates) = count_occurrences(&entity, question, &search_pool, scope_month);
            let year_label = {
                const MONTH_NAMES_DISP: [&str; 12] = [
                    "January","February","March","April","May","June",
                    "July","August","September","October","November","December",
                ];
                let month_part = scope_month
                    .map(|m| format!(" in {}", MONTH_NAMES_DISP[(m - 1) as usize]))
                    .unwrap_or_default();
                // When there is no month scope, add "in" before the year.
                let year_part = query_year.map(|y| {
                    if scope_month.is_none() { format!(" in {y}") }
                    else                     { format!(" {y}") }
                }).unwrap_or_default();
                format!("{month_part}{year_part}")
            };
            let entity_label = if entity.is_empty() {
                "the subject".to_string()
            } else {
                // Title-case the entity label for display.
                entity.split_whitespace()
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            None    => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            if count == 0 {
                println!(
                    "\n\x1b[33mAnswer\x1b[0m [count]: No occurrences of \"{entity_label}\" found{year_label}.\n"
                );
            } else {
                println!(
                    "\n\x1b[32;1mAnswer\x1b[0m [count]: \"{entity_label}\" appears \x1b[1m{count}\x1b[0m time(s){year_label}:"
                );
                let show = dates.len().min(25);
                for (i, d) in dates.iter().take(show).enumerate() {
                    println!("  {:2}. {}", i + 1, d);
                }
                if dates.len() > show {
                    println!("  … and {} more.", dates.len() - show);
                }
                println!();
            }
            continue; // skip the normal QA pipeline
        }

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
