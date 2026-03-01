/// word-doc-qa — Command-line QA system powered by a Burn transformer.
///
/// Usage:
///   word-doc-qa train  <docs_dir> [model_dir]
///   word-doc-qa infer  <docs_dir> [model_dir]
///
/// <docs_dir>  : directory containing .docx files
/// [model_dir] : where to read/write model checkpoints (default: ./model_output)

mod data;
mod model;
mod training;
mod inference;

use std::path::PathBuf;

fn usage() {
    eprintln!(
        "Usage:\n  word-doc-qa train  <docs_dir> [model_dir]\n  word-doc-qa infer  <docs_dir> [model_dir]"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        usage();
        std::process::exit(1);
    }

    let mode      = args[1].as_str();
    let docs_dir  = PathBuf::from(&args[2]);
    let model_dir = if args.len() >= 4 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from("model_output")
    };

    if !docs_dir.is_dir() {
        eprintln!("Error: '{}' is not a directory.", docs_dir.display());
        std::process::exit(1);
    }

    match mode {
        "train" => {
            let cfg = training::TrainingConfig::default();
            println!("[main] Training config: {cfg:?}\n");
            training::train(&docs_dir, &model_dir, &cfg);
        }
        "infer" | "inference" => {
            inference::run_inference(&docs_dir, &model_dir);
        }
        other => {
            eprintln!("Unknown mode '{}'. Expected 'train' or 'infer'.", other);
            usage();
            std::process::exit(1);
        }
    }
}
