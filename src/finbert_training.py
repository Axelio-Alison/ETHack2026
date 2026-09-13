"""Small, reusable training helpers for the direction model.

The notebook keeps configuration and interpretation visible.  The mechanical
training loop lives here so that it can be tested and reused without copying a
large block of code into the notebook.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

REQUIRED_COLUMNS = {
    "row_id",
    "story_group_id",
    "leakage_family_id",
    "ticker",
    "company_name",
    "pillar",
    "headline",
    "input_text",
    "final_direction",
    "label_id",
    "split",
}


@dataclass(frozen=True)
class TrainingConfig:
    """The checked recipe used for the selected experiment."""

    base_model: str = "ProsusAI/finbert"
    base_model_revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43"
    seeds: tuple[int, ...] = (17, 42, 73)
    max_epochs: int = 5
    patience: int = 2
    learning_rate: float = 2e-5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_length: int = 128
    warmup_ratio: float = 0.10
    gradient_clip: float = 1.0
    expected_split_rows: dict[str, int] = field(
        default_factory=lambda: {"train": 5730, "dev": 119, "test": 856}
    )


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_headline(text: Any) -> str:
    text = str(text).lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def validate_direction_dataset(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_split_rows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Load the labeled CSV and fail early on schema or split leakage."""

    path = Path(path)
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Training CSV does not match the checked SHA-256 digest.")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    if not frame["row_id"].is_unique:
        raise ValueError("row_id must be unique.")
    if set(frame["final_direction"]) != set(LABELS):
        raise ValueError(f"Expected exactly these labels: {list(LABELS)}")
    if not frame["label_id"].astype(int).equals(
        frame["final_direction"].map(LABEL_TO_ID).astype(int)
    ):
        raise ValueError("label_id does not match the documented label mapping.")

    expected_text = (
        "Company: "
        + frame["company_name"].astype(str)
        + ". Pillar: "
        + frame["pillar"].astype(str)
        + ". Headline: "
        + frame["headline"].astype(str)
    )
    if not frame["input_text"].equals(expected_text):
        raise ValueError("input_text does not match the documented template.")

    audit = frame.assign(normalized_headline=frame["headline"].map(_normalized_headline))
    for column in ("leakage_family_id", "story_group_id", "normalized_headline"):
        split_count = audit.groupby(column, dropna=False)["split"].nunique()
        if (split_count > 1).any():
            raise ValueError(f"A {column} family appears in more than one split.")
    if audit.duplicated(["ticker", "normalized_headline"]).any():
        raise ValueError("Duplicate target-company headlines remain in the dataset.")

    expected = expected_split_rows or TrainingConfig().expected_split_rows
    actual = frame["split"].value_counts().to_dict()
    if actual != expected:
        raise ValueError(f"Unexpected split sizes: {actual}; expected {expected}.")
    return frame


def _metric_block(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    label_ids = list(range(len(LABELS)))
    return {
        "rows": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=label_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=label_ids).tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=label_ids,
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }


def _class_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.loc[frame["split"].eq("train"), "final_direction"].value_counts()
    train_rows = int(counts.sum())
    return np.asarray(
        [np.sqrt(train_rows / (len(LABELS) * counts[label])) for label in LABELS],
        dtype="float32",
    )


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TokenizedRows:
    """Minimal dataset compatible with ``torch.utils.data.DataLoader``."""

    def __init__(self, rows: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.labels = rows["final_direction"].map(LABEL_TO_ID).astype(int).tolist()
        self.tokens = tokenizer(
            rows["input_text"].tolist(), truncation=True, max_length=max_length
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.tokens.items()}
        item["labels"] = self.labels[index]
        return item


def _make_loader(
    dataset: _TokenizedRows,
    *,
    batch_size: int,
    collator: Any,
    use_cuda: bool,
    shuffle: bool = False,
) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        pin_memory=use_cuda,
    )


def _predict_probabilities(
    model: Any,
    dataset: _TokenizedRows,
    *,
    collator: Any,
    device: Any,
    use_cuda: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    actual: list[int] = []
    probability_parts: list[np.ndarray] = []
    with torch.no_grad():
        loader = _make_loader(
            dataset,
            batch_size=batch_size,
            collator=collator,
            use_cuda=use_cuda,
        )
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_cuda
            ):
                logits = model(**inputs).logits
            actual.extend(labels.tolist())
            probability_parts.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.asarray(actual), np.concatenate(probability_parts)


def _train_one_epoch(
    model: Any,
    loader: Any,
    *,
    optimizer: Any,
    scheduler: Any,
    criterion: Any,
    scaler: Any,
    device: Any,
    use_cuda: bool,
    gradient_clip: float,
) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for batch in loader:
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run_seed_experiment(
    *,
    seed: int,
    datasets: dict[str, _TokenizedRows],
    collator: Any,
    device: Any,
    use_cuda: bool,
    class_weights: np.ndarray,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one seed and keep its best development state in CPU memory."""

    import torch
    from transformers import AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup

    _set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    train_loader = _make_loader(
        datasets["train"],
        batch_size=config.train_batch_size,
        collator=collator,
        use_cuda=use_cuda,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_dev = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        mean_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_cuda=use_cuda,
            gradient_clip=config.gradient_clip,
        )
        dev_actual, dev_probabilities = _predict_probabilities(
            model,
            datasets["dev"],
            collator=collator,
            device=device,
            use_cuda=use_cuda,
            batch_size=config.eval_batch_size,
        )
        dev_metrics = _metric_block(dev_actual, dev_probabilities.argmax(axis=1))
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "dev_macro_f1": dev_metrics["macro_f1"],
                "seconds": round(time.perf_counter() - epoch_started, 3),
            }
        )
        print(f"seed={seed} {history[-1]}")

        if dev_metrics["macro_f1"] > best_dev + 1e-6:
            best_dev = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No development checkpoint was selected.")
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "dev_macro_f1": best_dev,
        "seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }
    del model, optimizer, scheduler, scaler
    if use_cuda:
        torch.cuda.empty_cache()
    return summary, best_state


def _evaluate_by_pillar(
    test_rows: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    results = {}
    for pillar in ("financial", "environmental", "social", "governance"):
        mask = test_rows["pillar"].eq(pillar).to_numpy()
        if mask.any():
            results[pillar] = _metric_block(actual[mask], predicted[mask])
    return results


def run_direction_experiment(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Reproduce the checked multi-seed loop, then evaluate the test once."""

    import torch
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        BertTokenizer,
        DataCollatorWithPadding,
    )

    cfg = config or TrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("Training on", torch.cuda.get_device_name(0) if use_cuda else "CPU")

    tokenizer = BertTokenizer.from_pretrained(
        cfg.base_model, revision=cfg.base_model_revision
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if use_cuda else None,
        return_tensors="pt",
    )
    split_rows = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "dev", "test")
    }
    datasets = {
        split: _TokenizedRows(rows, tokenizer, cfg.max_length)
        for split, rows in split_rows.items()
    }
    weights = _class_weights(frame)

    seed_runs: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_state: dict[str, Any] | None = None
    for seed in cfg.seeds:
        run, state = run_seed_experiment(
            seed=seed,
            datasets=datasets,
            collator=collator,
            device=device,
            use_cuda=use_cuda,
            class_weights=weights,
            config=cfg,
        )
        seed_runs.append(run)
        if selected is None or run["dev_macro_f1"] > selected["dev_macro_f1"]:
            selected = run
            selected_state = state
        else:
            del state

    if selected is None or selected_state is None:
        raise RuntimeError("No seed was selected.")
    print(f"Selected seed {selected['seed']} using development macro-F1 only.")

    selected_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        revision=cfg.base_model_revision,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    selected_model.load_state_dict(selected_state)
    selected_model = selected_model.to(device)
    test_actual, test_probabilities = _predict_probabilities(
        selected_model,
        datasets["test"],
        collator=collator,
        device=device,
        use_cuda=use_cuda,
        batch_size=cfg.eval_batch_size,
    )
    test_predicted = test_probabilities.argmax(axis=1)
    test_metrics = _metric_block(test_actual, test_predicted)
    test_metrics["by_pillar"] = _evaluate_by_pillar(
        split_rows["test"], test_actual, test_predicted
    )

    row_numbers = split_rows["test"]["row_id"].str.slice(1).astype(int).to_numpy()
    challenge_mask = row_numbers > 6400
    challenge_metrics = _metric_block(
        test_actual[challenge_mask], test_predicted[challenge_mask]
    )
    challenge_metrics["by_pillar"] = _evaluate_by_pillar(
        split_rows["test"].loc[challenge_mask].reset_index(drop=True),
        test_actual[challenge_mask],
        test_predicted[challenge_mask],
    )

    model_dir = output_dir / "selected_model"
    selected_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    predictions = split_rows["test"][
        ["row_id", "ticker", "company_name", "pillar", "headline", "final_direction"]
    ].copy()
    for index, label in enumerate(LABELS):
        predictions[f"p_{label}"] = test_probabilities[:, index]
    predictions["predicted_direction"] = [
        ID_TO_LABEL[index] for index in test_predicted
    ]
    predictions["correct"] = predictions["predicted_direction"].eq(
        predictions["final_direction"]
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    result = {
        "base_model": cfg.base_model,
        "base_model_revision": cfg.base_model_revision,
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "selection_rule": "highest development macro-F1",
        "dataset_sha256": dataset_sha256,
        "split_rows": frame["split"].value_counts().to_dict(),
        "class_weights": {
            label: float(weight) for label, weight in zip(LABELS, weights)
        },
        "seed_development_runs": seed_runs,
        "combined_test": test_metrics,
        "active_challenge_test": challenge_metrics,
        "config": asdict(cfg),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    archive_path = Path(
        shutil.make_archive(
            str(output_dir / "finbert_direction_selected"), "zip", root_dir=model_dir
        )
    )
    checkpoint_sha256 = file_sha256(archive_path)
    (output_dir / "checkpoint_sha256.txt").write_text(
        f"{checkpoint_sha256}  {archive_path.name}\n", encoding="utf-8"
    )
    result["checkpoint_sha256"] = checkpoint_sha256
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
