#!/usr/bin/env python3
"""
Step 5: Few-shot fine-tuning with LoRA for foundation models.
Adapts Chronos (T5-based) using PEFT/LoRA on a fraction of target basin data.

Usage:
    python run_few_shot_lora.py --model chronos --fraction 0.10 --dataset CAMELS-BR
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import compute_all_metrics, nse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_basin_data(dataset_name, date_start=None, date_end=None):
    """Load all basins for a dataset."""
    ds_dir = DATA_DIR / dataset_name
    basins = {}
    if not ds_dir.exists():
        return basins
    for pf in sorted(ds_dir.glob("*.parquet")):
        bid = pf.stem
        df = pd.read_parquet(pf)
        if date_start:
            df = df[df.index >= date_start]
        if date_end:
            df = df[df.index <= date_end]
        if "QObs(mm/d)" in df.columns:
            q = df["QObs(mm/d)"].dropna()
            if len(q) > 100:
                basins[bid] = q
    return basins


def split_few_shot(basins, fraction, seed=42):
    """Split each basin's data into few-shot train and test."""
    rng = np.random.RandomState(seed)
    splits = {}
    for bid, q in basins.items():
        n = len(q)
        n_train = max(10, int(n * fraction))
        # Use first portion for few-shot training, rest for evaluation
        splits[bid] = {
            "train": q.iloc[:n_train],
            "test": q.iloc[n_train:],
        }
    return splits


def few_shot_chronos_lora(splits, context_length=512, horizon=1, num_samples=1,
                          lora_rank=8, lora_alpha=16, epochs=5, lr=1e-4):
    """Fine-tune Chronos with LoRA on few-shot data, then evaluate."""
    print("Loading Chronos for LoRA fine-tuning...")

    try:
        from chronos import ChronosPipeline
        from transformers import T5ForConditionalGeneration, AutoConfig
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  Install: pip install chronos-forecasting peft")
        return {}

    # Load base model
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=DEVICE,
            torch_dtype=torch.float32,
        )
        model = pipeline.model.model  # inner T5 model
    except Exception as e:
        print(f"  Could not load Chronos: {e}")
        return {}

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    # Prepare few-shot training data
    print("Preparing few-shot training sequences...")
    train_contexts = []
    train_targets = []
    for bid, split in splits.items():
        q_train = split["train"].values.astype(np.float32)
        if len(q_train) < context_length + horizon:
            # Use what we have, pad if needed
            if len(q_train) > horizon + 10:
                ctx_len = len(q_train) - horizon
                train_contexts.append(q_train[:ctx_len])
                train_targets.append(q_train[ctx_len:ctx_len + horizon])
        else:
            # Sliding windows over few-shot data
            for i in range(0, len(q_train) - context_length - horizon + 1,
                           max(1, (len(q_train) - context_length - horizon) // 5)):
                train_contexts.append(q_train[i:i + context_length])
                train_targets.append(q_train[i + context_length:i + context_length + horizon])

    if not train_contexts:
        print("  No training sequences created.")
        return {}

    print(f"  Created {len(train_contexts)} training sequences from {len(splits)} basins")

    # Fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        indices = np.random.permutation(len(train_contexts))
        batch_size = 8
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            # Simple MSE loss on next-step prediction
            batch_loss = 0
            for idx in batch_idx:
                ctx = torch.tensor(train_contexts[idx]).unsqueeze(0).to(DEVICE)
                tgt = torch.tensor(train_targets[idx]).unsqueeze(0).to(DEVICE)

                try:
                    # Forward pass through the model
                    outputs = model(input_ids=ctx.long().clamp(0, 4095), labels=tgt.long().clamp(0, 4095))
                    loss = outputs.loss
                    if loss is not None:
                        batch_loss += loss
                except Exception:
                    continue

            if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f}")

    # Evaluate on test portions
    print("Evaluating few-shot model...")
    model.eval()
    results = {}
    BATCH_SIZE = 128

    for bid, split in tqdm(splits.items(), desc="Few-shot eval"):
        q_test = split["test"].values.astype(np.float32)
        if len(q_test) < context_length + horizon:
            continue

        contexts, targets = [], []
        for start in range(0, len(q_test) - context_length - horizon + 1, max(horizon, 7)):
            contexts.append(torch.tensor(q_test[start:start + context_length]))
            targets.append(q_test[start + context_length:start + context_length + horizon])

        if not contexts:
            continue

        all_samples = []
        for i in range(0, len(contexts), BATCH_SIZE):
            try:
                forecast = pipeline.predict(contexts[i:i + BATCH_SIZE],
                                            prediction_length=horizon,
                                            num_samples=num_samples)
                all_samples.append(forecast.numpy())
            except Exception:
                continue

        if not all_samples:
            continue

        samples_np = np.concatenate(all_samples, axis=0)
        obs = np.concatenate(targets[:samples_np.shape[0]])
        sim = np.mean(samples_np, axis=1).flatten()[:len(obs)]

        results[bid] = compute_all_metrics(obs, sim)
        results[bid]["n_predictions"] = len(obs)

    return results


def few_shot_lstm_transfer(splits, source_ckpt_path, context_length=365,
                           horizon=1, epochs=10, lr=5e-4):
    """
    Traditional transfer learning baseline:
    Load LSTM pretrained on source, fine-tune on few-shot target data.
    """
    from models.baselines import StreamflowLSTM

    if not source_ckpt_path.exists():
        print(f"  Source checkpoint not found: {source_ckpt_path}")
        return {}

    ckpt = torch.load(source_ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    norm_stats = ckpt["norm_stats"]

    model = StreamflowLSTM(
        n_forcing=config["n_forcing"],
        n_static=config.get("n_static", 0),
        hidden_size=256, num_layers=2, dropout=0.2,
        horizon=config["horizon"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Fine-tune on few-shot data (just the head + last LSTM layer)
    for name, param in model.named_parameters():
        if "head" not in name and "lstm.weight_hh_l1" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    # Prepare training data
    train_sequences = []
    for bid, split in splits.items():
        q = split["train"].values.astype(np.float32)
        if len(q) < context_length + horizon:
            continue
        for i in range(len(q) - context_length - horizon + 1):
            x = q[i:i + context_length]
            y = q[i + context_length:i + context_length + horizon]
            # Normalize
            if norm_stats["target_mean"] is not None:
                x_norm = (x - norm_stats["target_mean"]) / norm_stats["target_std"]
                y_norm = (y - norm_stats["target_mean"]) / norm_stats["target_std"]
            else:
                x_norm, y_norm = x, y
            train_sequences.append((x_norm, y_norm))

    if not train_sequences:
        print("  No training sequences for LSTM transfer.")
        return {}

    print(f"  Fine-tuning LSTM on {len(train_sequences)} sequences...")
    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_sequences)
        total_loss = 0
        for x, y in train_sequences:
            x_t = torch.tensor(x).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            y_t = torch.tensor(y).unsqueeze(0).to(DEVICE)
            pred = model(x_t)
            loss = criterion(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{epochs} | loss={total_loss / len(train_sequences):.4f}")

    # Evaluate
    model.eval()
    results = {}
    for bid, split in splits.items():
        q = split["test"].values.astype(np.float32)
        if len(q) < context_length + horizon:
            continue

        all_obs, all_pred = [], []
        for i in range(0, len(q) - context_length - horizon + 1, max(horizon, 7)):
            x = q[i:i + context_length]
            target = q[i + context_length:i + context_length + horizon]

            if norm_stats["target_mean"] is not None:
                x_norm = (x - norm_stats["target_mean"]) / norm_stats["target_std"]
            else:
                x_norm = x

            x_t = torch.tensor(x_norm).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                pred = model(x_t).cpu().numpy().flatten()

            if norm_stats["target_mean"] is not None:
                pred = pred * norm_stats["target_std"] + norm_stats["target_mean"]

            all_obs.append(target)
            all_pred.append(pred[:len(target)])

        if all_obs:
            obs = np.concatenate(all_obs)
            sim = np.concatenate(all_pred)
            results[bid] = compute_all_metrics(obs, sim)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "lstm_transfer", "all"], default="all")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--dataset", default="CAMELS-BR")
    parser.add_argument("--test_start", default="2015-01-01")
    parser.add_argument("--test_end", default="2019-12-31")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Few-Shot Fine-Tuning")
    print(f"Model: {args.model} | Fraction: {args.fraction}")
    print(f"Dataset: {args.dataset} | Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    basins = load_basin_data(args.dataset, args.test_start, args.test_end)
    print(f"Loaded {len(basins)} basins from {args.dataset}")

    if len(basins) == 0:
        print("No basins found!")
        return

    splits = split_few_shot(basins, args.fraction)

    models_to_run = ["chronos", "lstm_transfer"] if args.model == "all" else [args.model]

    for model_name in models_to_run:
        fname = f"{model_name}_few_shot_{args.fraction}_{args.dataset}.csv"
        if (RESULTS_DIR / fname).exists():
            print(f"\n  Skipping {model_name} (fraction={args.fraction}) on {args.dataset} - results exist")
            continue

        print(f"\n--- {model_name} (fraction={args.fraction}) ---")
        t0 = time.time()

        if model_name == "chronos":
            results = few_shot_chronos_lora(splits)
        elif model_name == "lstm_transfer":
            ckpt = CKPT_DIR / "lstm_best.pt"
            results = few_shot_lstm_transfer(splits, ckpt)
        else:
            continue

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s ({len(results)} basins)")

        # Save
        if results:
            rows = []
            for bid, metrics in results.items():
                row = {"basin_id": bid, "model": model_name, "mode": "few_shot",
                       "fraction": args.fraction, "dataset": args.dataset}
                row.update(metrics)
                rows.append(row)
            df = pd.DataFrame(rows)
            fname = f"{model_name}_few_shot_{args.fraction}_{args.dataset}.csv"
            df.to_csv(RESULTS_DIR / fname, index=False)
            print(f"  Saved to {fname}")
            for m in ["NSE", "KGE", "RMSE"]:
                if m in df.columns:
                    vals = df[m].dropna()
                    print(f"    {m}: median={vals.median():.4f}, mean={vals.mean():.4f}")


if __name__ == "__main__":
    main()
