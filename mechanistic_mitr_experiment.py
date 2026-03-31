import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    train_samples: int = 8000
    val_samples: int = 1500
    contradiction_samples: int = 500
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.06
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    use_bf16: bool = True
    hidden_probe_epochs: int = 120
    hidden_probe_lr: float = 1e-2
    hidden_probe_wd: float = 1e-4
    probe_train_samples: int = 2500
    probe_val_samples: int = 1000
    mi_lambda: float = 0.01
    mi_warmup_steps: int = 200
    mi_strategy: str = "cosine"  # cosine or cka
    output_dir: str = "mechanistic_results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device(cfg: Config) -> Tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and cfg.use_bf16 and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    return device, dtype


def subsample(dataset, n: int, seed: int):
    if n < 0 or n >= len(dataset):
        return dataset
    rng = random.Random(seed)
    return dataset.select(rng.sample(range(len(dataset)), n))


def has_negation(question: str) -> int:
    q = question.lower()
    neg_markers = [" not ", "n't", " no ", " never ", " none ", " cannot ", " can't "]
    return int(any(marker in f" {q} " for marker in neg_markers))


def load_boolq(train_samples: int, val_samples: int, seed: int):
    ds = load_dataset("google/boolq")

    def fmt(ex):
        q = ex["question"].strip()
        return {
            "text": q + " [SEP] " + ex["passage"][:400],
            "label": int(ex["answer"]),
            "question": q,
            "negation_label": has_negation(q),
        }

    train = ds["train"].map(fmt, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(fmt, remove_columns=ds["validation"].column_names)
    train = subsample(train, train_samples, seed)
    val = subsample(val, val_samples, seed + 1)
    return train, val


_AUX = [
    ("is ", "is not "),
    ("are ", "are not "),
    ("was ", "was not "),
    ("were ", "were not "),
    ("does ", "does not "),
    ("do ", "do not "),
    ("did ", "did not "),
    ("has ", "has not "),
    ("have ", "have not "),
    ("had ", "had not "),
    ("can ", "cannot "),
    ("could ", "could not "),
    ("will ", "will not "),
    ("would ", "would not "),
    ("should ", "should not "),
]


def negate_question(q: str):
    q = q.strip().rstrip("?").lower()
    for pos, neg in _AUX:
        if q.startswith(neg):
            return pos + q[len(neg) :]
        if q.startswith(pos):
            return neg + q[len(pos) :]
    return None


def create_contradiction_pairs(dataset, n_pairs: int) -> List[Dict]:
    pairs = []
    for ex in dataset:
        q = ex.get("question", "").strip()
        q_neg = negate_question(q)
        if not q_neg or q_neg == q:
            continue

        text_fwd = ex["text"]
        # Keep the same passage/context and only negate the question.
        if "[SEP]" in text_fwd:
            parts = text_fwd.split("[SEP]", 1)
            passage = parts[1].strip()
            text_neg = q_neg + " [SEP] " + passage
        else:
            text_neg = q_neg

        pairs.append(
            {
                "text_forward": text_fwd,
                "text_negated": text_neg,
                "label_forward": ex["label"],
                "label_negated": 1 - ex["label"],
            }
        )
        if len(pairs) >= n_pairs:
            break
    return pairs


class LogicDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int):
        enc = tokenizer(
            [ex["text"] for ex in data],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([ex["label"] for ex in data], dtype=torch.long)
        self.negation_labels = torch.tensor(
            [ex["negation_label"] for ex in data], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "negation_labels": self.negation_labels[idx],
        }


class ContradictionPairDataset(Dataset):
    def __init__(self, pairs: Sequence[Dict], tokenizer, max_length: int):
        fwd = tokenizer(
            [p["text_forward"] for p in pairs],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg = tokenizer(
            [p["text_negated"] for p in pairs],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.fwd_ids = fwd["input_ids"]
        self.fwd_mask = fwd["attention_mask"]
        self.neg_ids = neg["input_ids"]
        self.neg_mask = neg["attention_mask"]
        self.fwd_lbl = torch.tensor([p["label_forward"] for p in pairs], dtype=torch.long)
        self.neg_lbl = torch.tensor([p["label_negated"] for p in pairs], dtype=torch.long)

    def __len__(self):
        return len(self.fwd_lbl)

    def __getitem__(self, idx):
        return {
            "fwd_input_ids": self.fwd_ids[idx],
            "fwd_attention_mask": self.fwd_mask[idx],
            "neg_input_ids": self.neg_ids[idx],
            "neg_attention_mask": self.neg_mask[idx],
            "fwd_label": self.fwd_lbl[idx],
            "neg_label": self.neg_lbl[idx],
        }


class CosineSimMI(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return (x_norm * y_norm).sum(dim=-1).mean().clamp(-10.0, 10.0)


class CKAMI(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cross = torch.norm(x.t() @ y) ** 2
        self_x = torch.norm(x.t() @ x)
        self_y = torch.norm(y.t() @ y)
        return (cross / (self_x * self_y + 1e-8)).clamp(-10.0, 10.0)


def build_pair_indices(num_diffs: int, pair_mode: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if pair_mode == "adjacent":
        for i in range(num_diffs - 1):
            pairs.append((i, i + 1))
        return pairs
    if pair_mode == "adjacent_plus_skip":
        for i in range(num_diffs - 1):
            pairs.append((i, i + 1))
        for i in range(num_diffs - 2):
            pairs.append((i, i + 2))
        return pairs
    if pair_mode == "all":
        for i in range(num_diffs):
            for j in range(i + 1, num_diffs):
                pairs.append((i, j))
        return pairs
    raise ValueError(f"Unsupported pair mode: {pair_mode}")


class BaselineClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        is_training=False,
        return_hidden=False,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden,
        )
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(F.relu(self.pre_classifier(cls)))
        logits = self.classifier(cls)

        result = {"logits": logits, "mi_loss": torch.tensor(0.0, device=logits.device)}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        if return_hidden:
            result["hidden_states"] = out.hidden_states
        return result


class MITRClassifier(nn.Module):
    def __init__(self, cfg: Config, pair_mode: str):
        super().__init__()
        self.cfg = cfg
        self.pair_mode = pair_mode
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden = self.encoder.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)
        self._step = 0
        self.mi_estimator = CosineSimMI() if cfg.mi_strategy == "cosine" else CKAMI()

    def _effective_lambda(self) -> float:
        if self._step >= self.cfg.mi_warmup_steps:
            return self.cfg.mi_lambda
        return self.cfg.mi_lambda * (self._step / max(1, self.cfg.mi_warmup_steps))

    def forward(self, input_ids, attention_mask, labels=None, is_training=False, return_hidden=False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(F.relu(self.pre_classifier(cls)))
        logits = self.classifier(cls)

        result = {"logits": logits}
        if return_hidden:
            result["hidden_states"] = out.hidden_states

        if labels is None:
            result["mi_loss"] = torch.tensor(0.0, device=logits.device)
            return result

        task_loss = F.cross_entropy(logits, labels)
        lam = self._effective_lambda()

        if is_training and lam > 0.0:
            hs = out.hidden_states
            diffs = []
            for i in range(len(hs) - 1):
                d = (hs[i + 1] - hs[i]).mean(dim=1)
                d = F.layer_norm(d, (d.size(-1),))
                diffs.append(d)

            pair_indices = build_pair_indices(len(diffs), self.pair_mode)
            mi_terms = [self.mi_estimator(diffs[i], diffs[j]) for i, j in pair_indices]
            if mi_terms:
                mi_mean = torch.stack(mi_terms).mean()
                result["loss"] = (1.0 - lam) * task_loss + lam * mi_mean
                result["mi_loss"] = mi_mean
            else:
                result["loss"] = task_loss
                result["mi_loss"] = torch.tensor(0.0, device=logits.device)
        else:
            result["loss"] = task_loss
            result["mi_loss"] = torch.tensor(0.0, device=logits.device)

        if is_training:
            self._step += 1

        return result


def build_optimizer_scheduler(model: nn.Module, train_loader, cfg: Config):
    no_decay = {"bias", "LayerNorm.weight"}
    grouped = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs // max(1, cfg.grad_accum)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, cfg: Config, device, dtype, is_mitr: bool):
    model.train()
    use_amp = dtype != torch.float32
    total_loss = 0.0
    total_mi = 0.0
    n = 0

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(ids, mask, labels=lbls, is_training=is_mitr)

        (out["loss"] / cfg.grad_accum).backward()

        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += out["loss"].item()
        total_mi += float(out["mi_loss"].detach().item())
        n += 1

    return {"train_loss": total_loss / n, "train_mi_loss": total_mi / n}


@torch.no_grad()
def eval_accuracy(model, loader, device, dtype):
    model.eval()
    use_amp = dtype != torch.float32
    correct = 0
    total = 0
    val_loss = 0.0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(ids, mask, labels=lbls)

        pred = out["logits"].argmax(dim=-1)
        correct += (pred == lbls).sum().item()
        total += lbls.size(0)
        val_loss += out["loss"].item()

    return {"accuracy": correct / total, "val_loss": val_loss / len(loader)}


@torch.no_grad()
def eval_contradiction(model, pair_loader, device, dtype):
    model.eval()
    use_amp = dtype != torch.float32
    contradictions = 0
    total = 0

    for batch in pair_loader:
        f_ids = batch["fwd_input_ids"].to(device)
        f_mask = batch["fwd_attention_mask"].to(device)
        n_ids = batch["neg_input_ids"].to(device)
        n_mask = batch["neg_attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            f_out = model(f_ids, f_mask)
            n_out = model(n_ids, n_mask)

        f_pred = f_out["logits"].argmax(dim=-1)
        n_pred = n_out["logits"].argmax(dim=-1)
        contradictions += (f_pred == n_pred).sum().item()
        total += f_pred.size(0)

    return {
        "contradiction_rate": contradictions / total,
        "consistency_rate": 1.0 - (contradictions / total),
    }


@torch.no_grad()
def extract_layer_cls_features(model, loader, device, dtype, max_samples: int):
    model.eval()
    use_amp = dtype != torch.float32
    layer_storage: Dict[int, List[np.ndarray]] = {}
    answer_labels: List[np.ndarray] = []
    neg_labels: List[np.ndarray] = []
    seen = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(ids, mask, return_hidden=True)

        hs = out["hidden_states"]
        for layer_idx, h in enumerate(hs):
            cls = h[:, 0, :].detach().float().cpu().numpy()
            layer_storage.setdefault(layer_idx, []).append(cls)

        answer_labels.append(batch["labels"].cpu().numpy())
        neg_labels.append(batch["negation_labels"].cpu().numpy())

        seen += batch["labels"].size(0)
        if seen >= max_samples:
            break

    merged = {k: np.concatenate(v, axis=0)[:max_samples] for k, v in layer_storage.items()}
    y_answer = np.concatenate(answer_labels, axis=0)[:max_samples]
    y_neg = np.concatenate(neg_labels, axis=0)[:max_samples]
    return merged, y_answer, y_neg


def train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)

    model = nn.Linear(xtr.size(1), 2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        logits = model(xtr)
        loss = F.cross_entropy(logits, ytr)
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        pred = model(xva).argmax(dim=-1)
        acc = (pred == yva).float().mean().item()
    return acc


def cka_linear(x: np.ndarray, y: np.ndarray) -> float:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    x_t = x_t - x_t.mean(dim=0, keepdim=True)
    y_t = y_t - y_t.mean(dim=0, keepdim=True)
    cross = torch.norm(x_t.t() @ y_t) ** 2
    self_x = torch.norm(x_t.t() @ x_t)
    self_y = torch.norm(y_t.t() @ y_t)
    return float((cross / (self_x * self_y + 1e-8)).item())


def compute_cka_matrix(layer_features: Dict[int, np.ndarray]) -> np.ndarray:
    layers = sorted(layer_features.keys())
    m = len(layers)
    mat = np.zeros((m, m), dtype=np.float32)
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            mat[i, j] = cka_linear(layer_features[li], layer_features[lj])
    return mat


def summarize_cka_band_stats(cka_mat: np.ndarray) -> Dict[str, float]:
    n = cka_mat.shape[0]
    adj = []
    non_adj = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == 1:
                adj.append(cka_mat[i, j])
            else:
                non_adj.append(cka_mat[i, j])
    return {
        "adjacent_mean": float(np.mean(adj)) if adj else 0.0,
        "non_adjacent_mean": float(np.mean(non_adj)) if non_adj else 0.0,
    }


def plot_training_histories(histories: Dict[str, Dict], output_dir: str):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for name, h in histories.items():
        plt.plot(h["epoch"], h["val_accuracy"], marker="o", label=name)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, h in histories.items():
        plt.plot(h["epoch"], h["train_mi_loss"], marker="o", label=name)
    plt.title("MI Proxy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MI")
    plt.legend()

    path = os.path.join(output_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_probe_profiles(
    layers: List[int],
    probe_map: Dict[str, Dict[str, List[float]]],
    task_name: str,
    output_path: str,
):
    plt.figure(figsize=(8, 4))
    for variant in probe_map:
        plt.plot(layers, probe_map[variant][task_name], marker="o", label=variant)
    plt.title(f"Layer Probe Accuracy: {task_name}")
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cka_heatmaps(cka_maps: Dict[str, np.ndarray], output_path: str):
    names = list(cka_maps.keys())
    plt.figure(figsize=(5 * len(names), 4))
    for i, name in enumerate(names, start=1):
        plt.subplot(1, len(names), i)
        mat = cka_maps[name]
        plt.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
        plt.title(name)
        plt.xlabel("Layer")
        plt.ylabel("Layer")
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_variant(
    variant: str,
    cfg: Config,
    train_loader,
    val_loader,
    pair_loader,
    device,
    dtype,
):
    if variant == "baseline":
        model = BaselineClassifier(cfg.model_name)
        is_mitr = False
    elif variant == "mitr_adj":
        model = MITRClassifier(cfg, pair_mode="adjacent")
        is_mitr = True
    elif variant == "mitr_all":
        model = MITRClassifier(cfg, pair_mode="all")
        is_mitr = True
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    model = model.to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, train_loader, cfg)
    history = {"epoch": [], "train_loss": [], "train_mi_loss": [], "val_accuracy": [], "val_loss": []}

    print(f"\n=== Running variant: {variant} ===")
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            cfg,
            device,
            dtype,
            is_mitr,
        )
        eval_metrics = eval_accuracy(model, val_loader, device, dtype)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_mi_loss"].append(train_metrics["train_mi_loss"])
        history["val_accuracy"].append(eval_metrics["accuracy"])
        history["val_loss"].append(eval_metrics["val_loss"])

        print(
            f"epoch={epoch} train={train_metrics['train_loss']:.4f} "
            f"mi={train_metrics['train_mi_loss']:.4f} val_acc={eval_metrics['accuracy']:.4f}"
        )

    contra = eval_contradiction(model, pair_loader, device, dtype)
    print(f"contradiction_rate={contra['contradiction_rate']:.4f}")

    return model, history, contra


def parse_args():
    parser = argparse.ArgumentParser(description="Mechanistic MITR evaluation in VS Code")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--val-samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--probe-train-samples", type=int, default=2500)
    parser.add_argument("--probe-val-samples", type=int, default=1000)
    parser.add_argument("--contradiction-samples", type=int, default=500)
    parser.add_argument("--mi-lambda", type=float, default=0.01)
    parser.add_argument("--mi-strategy", choices=["cosine", "cka"], default="cosine")
    parser.add_argument(
        "--variants",
        default="baseline,mitr_adj,mitr_all",
        help="Comma-separated subset of baseline,mitr_adj,mitr_all",
    )
    parser.add_argument("--output-dir", default="mechanistic_results")
    parser.add_argument("--seed", type=int, default=42)
    # In notebooks/ipykernel, extra args (e.g. "-f ...kernel.json") are injected.
    # parse_known_args keeps CLI behavior while ignoring unrelated notebook args.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    cfg = Config(
        model_name=args.model,
        epochs=args.epochs,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        probe_train_samples=args.probe_train_samples,
        probe_val_samples=args.probe_val_samples,
        contradiction_samples=args.contradiction_samples,
        mi_lambda=args.mi_lambda,
        mi_strategy=args.mi_strategy,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device, dtype = infer_device(cfg)

    print(f"device={device} dtype={dtype}")
    print("Loading BoolQ...")
    train_raw, val_raw = load_boolq(cfg.train_samples, cfg.val_samples, cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = LogicDataset(train_raw, tokenizer, cfg.max_length)
    val_ds = LogicDataset(val_raw, tokenizer, cfg.max_length)
    pairs = create_contradiction_pairs(val_raw, cfg.contradiction_samples)
    pair_ds = ContradictionPairDataset(pairs, tokenizer, cfg.max_length)

    dl_kw = {
        "pin_memory": device.type == "cuda",
        "num_workers": 2,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kw)
    pair_loader = DataLoader(pair_ds, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kw)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    histories: Dict[str, Dict] = {}
    final_metrics: Dict[str, Dict] = {}
    trained_models: Dict[str, nn.Module] = {}

    for variant in variants:
        set_seed(cfg.seed)
        model, history, contra = run_variant(
            variant,
            cfg,
            train_loader,
            val_loader,
            pair_loader,
            device,
            dtype,
        )
        trained_models[variant] = model
        histories[variant] = history
        final_metrics[variant] = {
            "final_accuracy": history["val_accuracy"][-1],
            "final_val_loss": history["val_loss"][-1],
            **contra,
        }

    # Mechanistic check 1: layer-wise probe profiles
    probe_results: Dict[str, Dict[str, List[float]]] = {}
    cka_maps: Dict[str, np.ndarray] = {}
    cka_stats: Dict[str, Dict[str, float]] = {}

    probe_train_loader = DataLoader(
        train_ds, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kw
    )
    probe_val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kw)

    for variant, model in trained_models.items():
        tr_layers, tr_ans, tr_neg = extract_layer_cls_features(
            model, probe_train_loader, device, dtype, cfg.probe_train_samples
        )
        va_layers, va_ans, va_neg = extract_layer_cls_features(
            model, probe_val_loader, device, dtype, cfg.probe_val_samples
        )

        layers = sorted(tr_layers.keys())
        answer_profile: List[float] = []
        negation_profile: List[float] = []

        for layer in layers:
            answer_acc = train_linear_probe(
                tr_layers[layer],
                tr_ans,
                va_layers[layer],
                va_ans,
                cfg.hidden_probe_epochs,
                cfg.hidden_probe_lr,
                cfg.hidden_probe_wd,
            )
            neg_acc = train_linear_probe(
                tr_layers[layer],
                tr_neg,
                va_layers[layer],
                va_neg,
                cfg.hidden_probe_epochs,
                cfg.hidden_probe_lr,
                cfg.hidden_probe_wd,
            )
            answer_profile.append(answer_acc)
            negation_profile.append(neg_acc)

        probe_results[variant] = {
            "answer": answer_profile,
            "negation": negation_profile,
            "layers": layers,
        }

        cka_mat = compute_cka_matrix(va_layers)
        cka_maps[variant] = cka_mat
        cka_stats[variant] = summarize_cka_band_stats(cka_mat)

    plot_training_histories(histories, cfg.output_dir)

    layer_axis = probe_results[variants[0]]["layers"]
    plot_probe_profiles(
        layer_axis,
        probe_results,
        "answer",
        os.path.join(cfg.output_dir, "probe_answer_profile.png"),
    )
    plot_probe_profiles(
        layer_axis,
        probe_results,
        "negation",
        os.path.join(cfg.output_dir, "probe_negation_profile.png"),
    )
    plot_cka_heatmaps(cka_maps, os.path.join(cfg.output_dir, "cka_heatmaps.png"))

    payload = {
        "config": asdict(cfg),
        "variants": variants,
        "final_metrics": final_metrics,
        "histories": histories,
        "probe_results": probe_results,
        "cka_stats": cka_stats,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved outputs:")
    print(f"- {cfg.output_dir}/training_curves.png")
    print(f"- {cfg.output_dir}/probe_answer_profile.png")
    print(f"- {cfg.output_dir}/probe_negation_profile.png")
    print(f"- {cfg.output_dir}/cka_heatmaps.png")
    print(f"- {cfg.output_dir}/summary.json")


if __name__ == "__main__":
    main()
