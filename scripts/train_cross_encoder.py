"""Fine-tune the cross-encoder NLI model on CLIMATE-FEVER splits."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune cross-encoder on CLIMATE-FEVER")
    p.add_argument("--model_name", default="climatebert/distilroberta-base-climate-f")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--output_dir", default="models/cross_encoder")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> Dataset:
    """Load a JSON split and convert to HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_fn(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["claim"],
        examples["evidence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading datasets...")
    train_ds = load_split(data_dir / "train.json")
    val_ds = load_split(data_dir / "val.json")

    logger.info("Tokenizing...")
    tok_fn = lambda ex: tokenize_fn(ex, tokenizer, args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Saving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training report
    report = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 2),
        "train_samples_per_second": round(
            train_result.metrics.get("train_samples_per_second", 0), 2
        ),
        "final_train_loss": round(train_result.metrics.get("train_loss", 0), 4),
    }

    report_path = Path("results") / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Training report saved to {report_path}")


if __name__ == "__main__":
    main()
