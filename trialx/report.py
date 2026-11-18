"""Render results/summary.csv as the Markdown table used in the README.

    python -m trialx.report                      # prints the table, writes results/summary.md
    python -m trialx.report --summary other.csv --out other.md

Copying numbers by hand from a CSV into a README is where transcription mistakes creep in.
This prints exactly what the benchmark wrote, with a fixed number of decimals per column.
Runs on different test-set sizes are flagged, because their scores are not comparable.
"""
import argparse
import csv
import os

# (summary.csv column, table header, decimal places)
TABLE_COLUMNS = [
    ("run_name", "Run", None),
    ("dtype", "Precision", None),
    ("overall", "Overall", 3),
    ("conditions_f1", "Conditions F1", 3),
    ("interventions_f1", "Interventions F1", 3),
    ("mean_seconds_per_example", "s / example", 2),
    ("p95_seconds_per_example", "p95 s", 2),
    ("tokens_per_second", "tokens/s", 1),
    ("peak_memory_mb", "Peak RAM (MB)", 0),
    ("model_size_mb", "Size (MB)", 0),
]


def format_cell(value, decimals):
    if value in (None, ""):
        return ""
    if decimals is None:
        return value
    try:
        return f"{float(value):.{decimals}f}"
    except ValueError:
        return value


def render_table(rows):
    lines = [
        "| " + " | ".join(header for _, header, _ in TABLE_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_COLUMNS),
    ]
    for row in rows:
        cells = [format_cell(row.get(column, ""), decimals) for column, _, decimals in TABLE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    example_counts = sorted({row.get("examples", "") for row in rows})
    if len(example_counts) > 1:
        lines.append("")
        lines.append(f"Warning: runs used different numbers of test examples ({', '.join(example_counts)}); "
                     "rerun with the same --limit before comparing scores.")
    elif example_counts:
        lines.append("")
        lines.append(f"All runs scored on {example_counts[0]} test examples.")
    return "\n".join(lines) + "\n"


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as summary_file:
        return list(csv.DictReader(summary_file))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=os.path.join("results", "summary.csv"))
    parser.add_argument("--out", default=os.path.join("results", "summary.md"))
    args = parser.parse_args(argv)

    if not os.path.isfile(args.summary):
        parser.error(f"{args.summary} not found; run the benchmarks first (scripts/run_benchmarks.sh)")
    rows = read_summary(args.summary)
    if not rows:
        parser.error(f"{args.summary} has no runs yet")

    table = render_table(rows)
    with open(args.out, "w", encoding="utf-8") as out_file:
        out_file.write(table)
    print(table, end="")


if __name__ == "__main__":
    main()
