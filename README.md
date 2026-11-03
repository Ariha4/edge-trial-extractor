# edge-trial-extractor

Fine-tune a small open LLM to pull structured facts out of neurotech clinical trial
descriptions, then quantize it and measure what that costs in accuracy and gains in
speed and memory when it runs on a laptop CPU instead of a GPU.

**Task.** Input: a trial's title, summary, and the start of its description from
ClinicalTrials.gov. Output: JSON with `study_type`, `intervention_types`,
`interventions`, and `conditions`.

**Why this task.** ClinicalTrials.gov already stores these fields in structured form,
so every trial comes with a free answer key. Scoring is automatic and exact, with no
paid LLM grader.

## Pipeline

```
ClinicalTrials.gov API ──> raw_studies.jsonl ──> train / val / test splits
                                                   │
             Qwen2.5-0.5B-Instruct ── LoRA fine-tune (Colab T4) ──> merged model
                                                   │
                         llama.cpp ── GGUF f16 ──> Q8_0 (8-bit), Q4_K_M (4-bit)
                                                   │
          benchmark each: accuracy (F1 per field), latency, tokens/s, peak memory, size
```

## Results

Filled in from `results/summary.csv` after running the benchmarks.

| Run | Precision | Overall | Conditions F1 | Interventions F1 | s / example | tokens/s | Peak RAM (MB) | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| Base model (no fine-tune) | fp16 | | | | | | | |
| Fine-tuned | fp16 | | | | | | | |
| Fine-tuned, llama.cpp | f16 | | | | | | | |
| Fine-tuned, llama.cpp | Q8_0 | | | | | | | |
| Fine-tuned, llama.cpp | Q4_K_M | | | | | | | |

## How to run

### 1. Set up (Mac)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

### 2. Build the dataset
```bash
python -m trialx.fetch --out data/raw_studies.jsonl --max-studies 4000
python -m trialx.build_dataset --raw data/raw_studies.jsonl --out-dir data
git add data/train.jsonl data/val.jsonl data/test.jsonl && git commit -m "Add dataset splits" && git push
```
Check the printed counts. Aim for at least 1,500 training examples. If you get fewer,
add search terms in `trialx/fetch.py` or raise `--max-studies`.

### 3. Fine-tune on Colab
Open `train_colab.ipynb` in Google Colab, set the runtime to T4 GPU, and run every
cell. The last cell downloads `ft-merged.zip`. Unzip it into `models/ft-merged/`.

### 4. Quantize (Mac)
```bash
brew install llama.cpp
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install -r ~/llama.cpp/requirements.txt
bash scripts/quantize.sh models/ft-merged ~/llama.cpp
```

### 5. Benchmark (Mac)
```bash
bash scripts/run_benchmarks.sh 20   # smoke test on 20 examples
bash scripts/run_benchmarks.sh      # full test set
```

## Design notes

- **Grounding filter.** A condition or intervention label is kept only if it appears
  in the input text. Without this, the model is graded on facts it cannot see.
- **Stable splits.** Each trial's split comes from a hash of its NCT ID, so rebuilding
  the dataset never leaks test trials into training.
- **Failures count.** An output that is not valid JSON is scored as an empty
  prediction, not skipped, so broken outputs lower the score instead of hiding.
- **Loss on the answer only.** Training uses prompt/completion pairs, so the model is
  trained to produce the JSON, not to reproduce the trial text.
- **Same prompt everywhere.** `trialx/prompts.py` is shared by training and evaluation.

## Stretch goals
- Run the Q4_K_M model on an iPhone with a llama.cpp-based app and record latency.
- Profile the model on Snapdragon hardware with Qualcomm AI Hub, if it supports this
  architecture.

## License
MIT
