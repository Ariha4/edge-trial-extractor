"""LoRA fine-tune of a small instruct model on the trial extraction task.

Run on a free Google Colab T4 GPU (see README, step 3). Takes roughly 10-20 minutes
for ~3k examples and 2 epochs; your times will vary.

What LoRA does here: the 0.5B base weights stay frozen. Small low-rank matrices
(rank 16) are added next to every linear layer, and only those are trained. That
is a few million trainable parameters instead of 500 million, which is why it
fits on a free GPU. At the end the adapters are merged back into the base
weights so the result is a normal model that llama.cpp can convert and quantize.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from trialx.prompts import build_messages, format_target  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_split(path):
    """Convert examples into TRL's conversational prompt/completion format.

    With prompt/completion data, the trainer computes loss only on the completion
    (the JSON answer), not on the trial text the model is reading.
    """
    rows = []
    with open(path, encoding="utf-8") as in_file:
        for line in in_file:
            example = json.loads(line)
            rows.append({
                "prompt": build_messages(example["text"]),
                "completion": [{"role": "assistant", "content": format_target(example["labels"])}],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="models/ft-merged")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # Load in float32 and let fp16 mixed precision handle speed. Loading the base
    # model directly in fp16 makes the optimizer fail on a T4.
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)

    train_dataset = Dataset.from_list(load_split(args.train))
    val_dataset = Dataset.from_list(load_split(args.val))
    print(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_config = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        max_length=1024,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
