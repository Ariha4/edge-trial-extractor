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

Every run writes results/<run-name>.json (metrics + raw outputs) and appends a
row to results/summary.csv.
"""
import argparse
import csv
import json
import os
import platform
import resource
import statistics
import time

from trialx.prompts import build_messages
from trialx.scoring import score

MAX_NEW_TOKENS = 200


def load_examples(path, limit):
    examples = []
    with open(path, encoding="utf-8") as in_file:
        for line in in_file:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
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


def run(args):
    examples = load_examples(args.data, args.limit)

    load_start = time.perf_counter()
    if args.backend == "hf":
        runner = HuggingFaceRunner(args.model, args.dtype, args.device)
    else:
        runner = GgufRunner(args.model, args.threads)
    load_seconds = time.perf_counter() - load_start

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
            print(f"{index + 1}/{len(examples)} done")

    references = []
    for example in examples:
        references.append(example["labels"])
    metrics = score(raw_outputs, references)

    total_generation_seconds = sum(seconds_per_example)
    metrics.update({
        "run_name": args.run_name,
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype if args.backend == "hf" else "gguf",
        "load_seconds": load_seconds,
        "mean_seconds_per_example": statistics.mean(seconds_per_example),
        "median_seconds_per_example": statistics.median(seconds_per_example),
        "tokens_per_second": total_generated_tokens / total_generation_seconds if total_generation_seconds else 0.0,
        "peak_memory_mb": peak_memory_mb(),
        "model_size_mb": model_size_mb(args.model),
        "machine": platform.platform(),
    })

    os.makedirs("results", exist_ok=True)
    details = {"metrics": metrics, "outputs": []}
    for example, raw_output in zip(examples, raw_outputs):
        details["outputs"].append({"nct_id": example["nct_id"], "output": raw_output, "labels": example["labels"]})
    with open(os.path.join("results", f"{args.run_name}.json"), "w", encoding="utf-8") as out_file:
        json.dump(details, out_file, indent=2, ensure_ascii=False)

    summary_path = os.path.join("results", "summary.csv")
    summary_columns = [
        "run_name", "backend", "dtype", "examples", "overall", "valid_json_rate", "study_type_accuracy",
        "intervention_types_f1", "interventions_f1", "conditions_f1", "mean_seconds_per_example",
        "tokens_per_second", "peak_memory_mb", "model_size_mb",
    ]
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    for key in summary_columns:
        print(f"{key}: {metrics.get(key)}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backend", choices=["hf", "gguf"], required=True)
    parser.add_argument("--model", required=True, help="Hub id, local HF folder, or .gguf file")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all test examples")
    parser.add_argument("--dtype", default="float16", help="hf backend only: float32 or float16")
    parser.add_argument("--device", default="cpu", help="hf backend only: cpu, mps, or cuda")
    parser.add_argument("--threads", type=int, default=4, help="gguf backend only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
