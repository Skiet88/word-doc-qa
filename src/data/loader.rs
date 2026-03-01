/// Load and extract plain text from .docx files using docx-rs.
use std::fs;
use std::path::Path;
use docx_rs::{read_docx, DocumentChild, DrawingData, ParagraphChild, RunChild, TableChild, TableRowChild, TableCellContent, TextBoxContentChild};

/// Upper-case month names used by inference to detect month-year prefixes.
pub const MONTH_NAMES: &[&str] = &[
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
];

#[derive(Debug, Clone)]
pub struct Document {
    pub filename: String,
    /// Calendar year parsed from the filename (e.g. 2025), if present.
    pub year: Option<u32>,
    /// Each element is one paragraph (non-empty) from the document.
    pub paragraphs: Vec<String>,
}

/// Extract text from a single run child element.
/// Drawing (WPS text-box) runs are handled separately in `drawing_cell_text`
/// so that each cell's day-number text always precedes the event label.
fn run_child_text(rc: &RunChild) -> String {
    match rc {
        RunChild::Text(t) => t.text.clone(),
        _ => String::new(),
    }
}

/// Extract all floating text-box labels from a table cell's paragraph runs.
/// These are WPS shapes (RunChild::Drawing → DrawingData::TextBox) used for
/// event-label bars in academic calendar documents.
fn drawing_cell_text(cell: &docx_rs::TableCell) -> String {
    let mut parts: Vec<String> = Vec::new();
    for cc in &cell.children {
        if let TableCellContent::Paragraph(p) = cc {
            for pc in &p.children {
                if let ParagraphChild::Run(run) = pc {
                    for rc in &run.children {
                        if let RunChild::Drawing(d) = rc {
                            if let Some(DrawingData::TextBox(tb)) = &d.data {
                                for child in &tb.children {
                                    if let TextBoxContentChild::Paragraph(tp) = child {
                                        let t = para_text(tp);
                                        let t = t.trim().to_string();
                                        if !t.is_empty() {
                                            parts.push(t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    parts.join(" ")
}

/// Extract text from a single paragraph child element.
fn para_child_text(c: &ParagraphChild) -> String {
    match c {
        ParagraphChild::Run(run) => run.children.iter().map(run_child_text).collect(),
        _ => String::new(),
    }
}

/// Extract text from a paragraph element.
fn para_text(p: &docx_rs::Paragraph) -> String {
    p.children.iter().map(para_child_text).collect()
}

/// Recursively extract text from table cells.
fn table_text(table: &docx_rs::Table) -> Vec<String> {
    let mut rows = Vec::new();
    for row_child in &table.rows {
        let TableChild::TableRow(row) = row_child;
        let mut cells: Vec<String> = Vec::new();
        for cell_child in &row.cells {
            let TableRowChild::TableCell(cell) = cell_child;

            // Regular (non-drawing) cell text — day numbers and inline events.
            let regular_text: String = cell
                .children
                .iter()
                .filter_map(|cc| match cc {
                    TableCellContent::Paragraph(p) => {
                        let t = para_text(p);
                        if t.trim().is_empty() { None } else { Some(t) }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");

            // Floating WPS text-box labels (e.g. "AUTUMN GRADUATION" bars).
            let drawing_text = drawing_cell_text(cell);

            // Build the combined cell string: day-number first, then event label.
            let cell_text = match (regular_text.trim(), drawing_text.trim()) {
                ("", "") => String::new(),
                ("", d) => d.to_string(),
                (r, "") => r.to_string(),
                (r, d) => format!("{} {}", r, d),
            };

            if !cell_text.trim().is_empty() {
                cells.push(cell_text.trim().to_string());
            }
        }
        if !cells.is_empty() {
            rows.push(cells.join(" | "));
        }
    }
    rows
}

/// Load a .docx file and extract its text as a `Document`.
pub fn load_docx(path: &Path) -> Result<Document, String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    let docx = read_docx(&data).map_err(|e| format!("{:?}", e))?;

    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Extract a 4-digit year (1900-2199) from the filename, e.g. "calendar_2025.docx" → 2025.
    let year: Option<u32> = {
        let mut buf = String::new();
        let mut found = None;
        for c in filename.chars() {
            if c.is_ascii_digit() {
                buf.push(c);
                if buf.len() == 4 {
                    if let Ok(y) = buf.parse::<u32>() {
                        if (1900..=2199).contains(&y) {
                            found = Some(y);
                            break;
                        }
                    }
                    buf.clear();
                }
            } else {
                buf.clear();
            }
        }
        found
    };

    let mut paragraphs: Vec<String> = Vec::new();

    for child in &docx.document.children {
        match child {
            DocumentChild::Paragraph(p) => {
                let text = para_text(p);
                let trimmed = text.trim().to_string();
                if !trimmed.is_empty() {
                    paragraphs.push(trimmed);
                }
            }
            DocumentChild::Table(t) => {
                // Represent each table row as a pseudo-paragraph so BERT can
                // answer questions about structured table content.
                for row in table_text(t) {
                    paragraphs.push(row);
                }
            }
            _ => {}
        }
    }

    Ok(Document { filename, year, paragraphs })
}

/// Load all .docx files found in `dir`.
pub fn load_all_docx(dir: &Path) -> Vec<Document> {
    let mut docs = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Cannot read directory {:?}: {}", dir, e);
            return docs;
        }
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().map(|ext| ext == "docx").unwrap_or(false) {
            match load_docx(&path) {
                Ok(doc) => {
                    println!(
                        "[loader] Loaded '{}' — {} paragraph(s)",
                        doc.filename,
                        doc.paragraphs.len()
                    );
                    docs.push(doc);
                }
                Err(e) => eprintln!("[loader] Warning: could not load {:?}: {}", path, e),
            }
        }
    }
    docs
}
