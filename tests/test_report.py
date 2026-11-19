import csv

import pytest

from trialx.evaluate import SUMMARY_COLUMNS, write_summary_row
from trialx.report import main, render_table


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as summary_file:
        return list(csv.DictReader(summary_file))


def test_rerun_replaces_row_in_place(tmp_path):
    summary_path = str(tmp_path / "summary.csv")
    write_summary_row(summary_path, {"run_name": "base", "examples": 20, "overall": 0.1})
    write_summary_row(summary_path, {"run_name": "ft", "examples": 20, "overall": 0.2})
    write_summary_row(summary_path, {"run_name": "base", "examples": 332, "overall": 0.3})

    rows = read_rows(summary_path)
    assert [(row["run_name"], row["examples"]) for row in rows] == [("base", "332"), ("ft", "20")]
    assert list(rows[0].keys()) == SUMMARY_COLUMNS
    assert not (tmp_path / "summary.csv.tmp").exists()


def test_old_summary_with_fewer_columns_is_upgraded(tmp_path):
    summary_path = tmp_path / "summary.csv"
    summary_path.write_text("run_name,overall\nold,0.5\n", encoding="utf-8")

    write_summary_row(str(summary_path), {"run_name": "new", "overall": 0.6, "git_commit": "abc123"})

    rows = read_rows(summary_path)
    assert rows[0]["run_name"] == "old" and rows[0]["git_commit"] == ""
    assert rows[1]["git_commit"] == "abc123"


def test_table_formats_numbers_and_leaves_missing_values_blank():
    table = render_table([{"run_name": "ft_q4", "dtype": "gguf", "examples": "332", "overall": "0.81234",
                           "tokens_per_second": "41.26", "model_size_mb": ""}])
    row = table.splitlines()[2]
    assert row.startswith("| ft_q4 | gguf | 0.812 |")
    assert "| 41.3 |" in row
    assert row.endswith("|  |")
    assert "All runs scored on 332 test examples." in table


def test_table_warns_when_runs_used_different_test_sizes():
    table = render_table([{"run_name": "a", "examples": "20"}, {"run_name": "b", "examples": "332"}])
    assert "different numbers of test examples (20, 332)" in table


def test_cli_writes_markdown(tmp_path, capsys):
    summary_path = str(tmp_path / "summary.csv")
    write_summary_row(summary_path, {"run_name": "base", "dtype": "float16", "examples": 2, "overall": 0.5})
    out_path = tmp_path / "summary.md"

    main(["--summary", summary_path, "--out", str(out_path)])

    assert out_path.read_text(encoding="utf-8") == capsys.readouterr().out
    assert "| base | float16 | 0.500 |" in out_path.read_text(encoding="utf-8")


def test_cli_explains_missing_summary(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--summary", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "out.md")])
    assert "run the benchmarks first" in capsys.readouterr().err
