//! Compare training metrics between two runs for regression detection.
//!
//! Usage:
//!   cargo run --example compare_metrics -- --baseline baseline.jsonl --current current.jsonl --tolerance 0.01

use anyhow::{Context, Result};
use clap::Parser;
use psyche_modeling::metrics::{compare_metrics, load_metrics, MetricsComparison};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "compare_metrics")]
#[command(about = "Compare training metrics between two runs for regression detection")]
struct Args {
    /// Path to the baseline metrics file (JSONL)
    #[arg(long)]
    baseline: PathBuf,

    /// Path to the current metrics file (JSONL)
    #[arg(long)]
    current: PathBuf,

    /// Maximum allowed loss deviation (absolute)
    #[arg(long, default_value_t = 0.01)]
    tolerance: f64,

    /// Output comparison as JSON
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn print_comparison(comparison: &MetricsComparison) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           METRICS COMPARISON REPORT                       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Steps compared:           {:>30} ║",
        comparison.steps_compared
    );
    println!(
        "║ Tolerance:                {:>30.6} ║",
        comparison.tolerance
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Max loss difference:      {:>30.6} ║",
        comparison.max_loss_diff
    );
    println!(
        "║ Max diff at step:         {:>30} ║",
        comparison.max_diff_step
    );
    println!(
        "║ Max diff percentage:      {:>29.2}% ║",
        comparison.max_loss_diff_pct
    );
    println!(
        "║ Mean loss difference:     {:>30.6} ║",
        comparison.mean_loss_diff
    );
    println!(
        "║ Final loss difference:    {:>30.6} ║",
        comparison.final_loss_diff
    );
    println!("╠══════════════════════════════════════════════════════════╣");

    let status = if comparison.passed {
        "✓ PASSED"
    } else {
        "✗ FAILED"
    };
    let color = if comparison.passed {
        "\x1b[32m"
    } else {
        "\x1b[31m"
    };
    println!(
        "║ Status:                   {}{:>30}\x1b[0m ║",
        color, status
    );
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load baseline metrics
    let baseline = load_metrics(&args.baseline)
        .with_context(|| format!("Failed to load baseline metrics from {:?}", args.baseline))?;

    if baseline.is_empty() {
        anyhow::bail!("Baseline metrics file is empty");
    }

    // Load current metrics
    let current = load_metrics(&args.current)
        .with_context(|| format!("Failed to load current metrics from {:?}", args.current))?;

    if current.is_empty() {
        anyhow::bail!("Current metrics file is empty");
    }

    println!(
        "Baseline: {} steps from {:?}",
        baseline.len(),
        args.baseline
    );
    println!("Current:  {} steps from {:?}", current.len(), args.current);
    println!();

    // Compare
    let comparison = compare_metrics(&baseline, &current, args.tolerance);

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&comparison)
                .context("Failed to serialize comparison to JSON")?
        );
    } else {
        print_comparison(&comparison);

        // Print loss trajectory comparison for key steps
        let key_steps = [0, 10, 50, 100, 500, 1000];
        println!();
        println!("Loss trajectory comparison:");
        println!("┌──────────┬────────────────┬────────────────┬────────────────┐");
        println!("│   Step   │    Baseline    │    Current     │     Diff       │");
        println!("├──────────┼────────────────┼────────────────┼────────────────┤");

        for &target_step in &key_steps {
            let baseline_entry = baseline.iter().find(|m| m.step == target_step);
            let current_entry = current.iter().find(|m| m.step == target_step);

            match (baseline_entry, current_entry) {
                (Some(b), Some(c)) => {
                    let diff = c.loss - b.loss;
                    let sign = if diff >= 0.0 { "+" } else { "" };
                    println!(
                        "│ {:>8} │ {:>14.6} │ {:>14.6} │ {:>13}{:.6} │",
                        target_step, b.loss, c.loss, sign, diff
                    );
                }
                (Some(b), None) => {
                    println!(
                        "│ {:>8} │ {:>14.6} │ {:>14} │ {:>14} │",
                        target_step, b.loss, "N/A", "N/A"
                    );
                }
                (None, Some(c)) => {
                    println!(
                        "│ {:>8} │ {:>14} │ {:>14.6} │ {:>14} │",
                        target_step, "N/A", c.loss, "N/A"
                    );
                }
                (None, None) => {}
            }
        }

        println!("└──────────┴────────────────┴────────────────┴────────────────┘");
    }

    // Exit with error code if comparison failed
    if !comparison.passed {
        std::process::exit(1);
    }

    Ok(())
}
