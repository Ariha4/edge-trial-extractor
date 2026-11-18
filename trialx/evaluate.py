"""Benchmark one model configuration: accuracy, latency, and memory.

Run each configuration in its own process so peak memory is measured per model.

Examples:
    # Base model, no fine-tuning (zero-shot baseline), full precision on the Mac GPU
    python -m trialx.evaluate --run-name base_fp16 --backend hf \
        --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --device mps

    # Fine-tuned model, still a Hugging Face checkpoint
    python -m trialx.evaluate --run-name ft_fp16 --backend hf \
        --model models/ft-merged --dtype float16 --device mps

    # Fine-tuned + quantized GGUF files run with llama.cpp (CPU, like a phone would)
    python -m trialx.evaluate --run-name ft_q8 --backend gguf --model models/ft-q8_0.gguf
    python -m trialx.evaluate --run-name ft_q4 --backend gguf --model models/ft-q4_k_m.gguf

Every run writes results/<run-name>.json (metrics, run metadata, raw outputs) and
writes its row in results/summary.csv. Before timing starts, `--warmup` examples
are generated and thrown away so one-time startup cost does not inflate latency.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from importlib import metadata

from trialx.build_dataset import VALID_STUDY_TYPES
from trialx.prompts import build_messages
from trialx.scoring import LIST_FIELDS, score

MAX_NEW_TOKENS = 200


class DataError(ValueError):
    """A test example is malformed. Raised before any model is loaded."""


def validate_example(example, where):
    """Check one example has the shape scoring expects, so a bad line fails fast with its location."""
    if not isinstance(example, dict):
        raise DataError(f"{where}: expected a JSON object, got {type(example).__name__}")
    for key in ("nct_id", "text"):
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise DataError(f"{where}: '{key}' must be a non-empty string")
    labels = example.get("labels")
    if not isinstance(labels, dict):
        raise DataError(f"{where}: 'labels' must be an object")
    if labels.get("study_type") not in VALID_STUDY_TYPES:
        raise DataError(f"{where}: labels.study_type must be one of {sorted(VALID_STUDY_TYPES)}, "
                        f"got {labels.get('study_type')!r}")
    for field_name in LIST_FIELDS:
        value = labels.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DataError(f"{where}: labels.{field_name} must be a list of strings")


def load_examples(path, limit=0):
    """Read and validate up to `limit` examples (0 = all) from a JSONL file."""
    if limit < 0:
        raise ValueError("limit must be 0 (all examples) or a positive number")
    examples = []
    with open(path, encoding="utf-8") as in_file:
        for line_number, line in enumerate(in_file, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"{where}: not valid JSON ({error.msg})") from None
            validate_example(example, where)
            examples.append(example)
            if limit and len(examples) >= limit:
                break
    if not examples:
        raise DataError(f"{path}: no examples found")
    return examples


def peak_memory_mb():
    """Peak resident memory of this process. macOS reports bytes, Linux reports kilobytes."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return max_rss / (1024 * 1024)
    return max_rss / 1024


def model_size_mb(model_path):
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    if os.path.isdir(model_path):
        total_bytes = 0
        for root, _, file_names in os.walk(model_path):
            for file_name in file_names:
                if file_name.endswith((".safetensors", ".bin")):
                    total_bytes += os.path.getsize(os.path.join(root, file_name))
        return total_bytes / (1024 * 1024)
    return None  # a Hub model id that is not downloaded locally


class HuggingFaceRunner:
    def __init__(self, model_path, dtype_name, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        torch_dtype = getattr(torch, dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
        self.model.eval()

    def generate(self, messages):
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return text, len(new_token_ids)


class GgufRunner:
    def __init__(self, model_path, threads):
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=threads, verbose=False)

    def generate(self, messages):
        response = self.llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        text = response["choices"][0]["message"]["content"]
        return text, response["usage"]["completion_tokens"]


SUMMARY_COLUMNS = [
    "run_name", "backend", "dtype", "examples", "overall", "valid_json_rate", "study_type_accuracy",
    "intervention_types_f1", "interventions_f1", "conditions_f1", "mean_seconds_per_example",
    "p95_seconds_per_example", "tokens_per_second", "peak_memory_mb", "model_size_mb", "git_commit",
]


def percentile(values, fraction):
    """Nearest-rank percentile: the smallest value with at least `fraction` of values at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def time_generations(runner, examples, warmup=0, progress=print):
    """Generate an answer for every example and time each call.

    `runner` is anything with generate(messages) -> (text, generated_token_count). The first
    `warmup` examples are generated once without timing, so first-call costs (memory allocation,
    kernel compilation, cache setup) are not counted as per-example latency.
    """
    for example in examples[:warmup]:
        runner.generate(build_messages(example["text"]))

    raw_outputs = []
    seconds_per_example = []
    total_generated_tokens = 0
    for index, example in enumerate(examples):
        start = time.perf_counter()
        text, generated_tokens = runner.generate(build_messages(example["text"]))
        seconds_per_example.append(time.perf_counter() - start)
        raw_outputs.append(text)
        total_generated_tokens += generated_tokens
        if (index + 1) % 25 == 0:
            progress(f"{index + 1}/{len(examples)} done")

    total_seconds = sum(seconds_per_example)
    timing = {
        "mean_seconds_per_example": statistics.mean(seconds_per_example),
        "median_seconds_per_example": statistics.median(seconds_per_example),
        "p95_seconds_per_example": percentile(seconds_per_example, 0.95),
        # Includes prompt processing, so this is end-to-end throughput, not pure decode speed.
        "tokens_per_second": total_generated_tokens / total_seconds if total_seconds else 0.0,
        "generated_tokens": total_generated_tokens,
    }
    return raw_outputs, timing


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as in_file:
        for block in iter(lambda: in_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    """(commit, dirty) for the working tree, or (None, None) outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip())


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_metadata(args):
    """Everything needed to tell whether two result files are comparable, or to rerun one."""
    commit, dirty = git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "data_path": args.data,
        "data_sha256": file_sha256(args.data),
        "limit": args.limit,
        "warmup": args.warmup,
        "threads": args.threads if args.backend == "gguf" else None,
        "device": args.device if args.backend == "hf" else "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "transformers", "llama_cpp_python")},
        "command": sys.argv,
    }


def write_summary_row(summary_path, row):
    """Insert this run's row, replacing any earlier row with the same run_name.

    Rerunning one configuration (say, after a smoke test with --limit 20) updates its row in place
    instead of appending a duplicate that a reader could mistake for the real result. The file is
    rewritten through a temp file so an interrupted write cannot truncate earlier results.
    """
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as summary_file:
            rows = list(csv.DictReader(summary_file))
    new_row = {column: row.get(column, "") for column in SUMMARY_COLUMNS}
    for index, existing in enumerate(rows):
        if existing.get("run_name") == new_row["run_name"]:
            rows[index] = new_row
            break
    else:
        rows.append(new_row)

    temp_path = summary_path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, summary_path)


def load_runner(args):
    if args.backend == "hf":
        return HuggingFaceRunner(args.model, args.dtype, args.device)
    return GgufRunner(args.model, args.threads)


def run(args, runner=None):
    """Benchmark one configuration. `runner` can be passed in to skip model loading (used by tests)."""
    examples = load_examples(args.data, args.limit)
    run_info = run_metadata(args)

    load_start = time.perf_counter()
    if runner is None:
        runner = load_runner(args)
    load_seconds = time.perf_counter() - load_start

    raw_outputs, timing = time_generations(runner, examples, args.warmup)
    metrics = score(raw_outputs, [example["labels"] for example in examples])
    metrics.update(timing)
    metrics.update({
        "run_name": args.run_name,
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype if args.backend == "hf" else "gguf",
        "load_seconds": load_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "model_size_mb": model_size_mb(args.model),
        "git_commit": (run_info["git_commit"] or "")[:12],
    })

    os.makedirs(args.results_dir, exist_ok=True)
    details = {"metrics": metrics, "run": run_info, "outputs": []}
    for example, raw_output in zip(examples, raw_outputs):
        details["outputs"].append({"nct_id": example["nct_id"], "output": raw_output, "labels": example["labels"]})
    with open(os.path.join(args.results_dir, f"{args.run_name}.json"), "w", encoding="utf-8") as out_file:
        json.dump(details, out_file, indent=2, ensure_ascii=False)

    write_summary_row(os.path.join(args.results_dir, "summary.csv"), metrics)

    for key in SUMMARY_COLUMNS:
        print(f"{key}: {metrics.get(key)}")
    return metrics


def non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backend", choices=["hf", "gguf"], required=True)
    parser.add_argument("--model", required=True, help="Hub id, local HF folder, or .gguf file")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--limit", type=non_negative_int, default=0, help="0 = all test examples")
    parser.add_argument("--warmup", type=non_negative_int, default=1,
                        help="untimed generations before timing starts")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"],
                        help="hf backend only")
    parser.add_argument("--device", default="cpu", help="hf backend only: cpu, mps, or cuda")
    parser.add_argument("--threads", type=positive_int, default=4, help="gguf backend only")
    parser.add_argument("--results-dir", default="results")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "gguf" and not os.path.isfile(args.model):
        parser.error(f"--model {args.model} is not a file (the gguf backend needs a .gguf path)")
    if not os.path.isfile(args.data):
        parser.error(f"--data {args.data} does not exist")
    try:
        run(args)
    except DataError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
