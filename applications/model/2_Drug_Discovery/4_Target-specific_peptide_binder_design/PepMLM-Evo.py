from transformers import Trainer, TrainingArguments, AutoTokenizer, EsmForMaskedLM
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.optim import AdamW
import numpy as np
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ================== 1. 定义命令行参数 ==================
# !!! ABSOLUTELY CANNOT CHANGE !!!
parser = argparse.ArgumentParser(description="PepMLM Training Script")
parser.add_argument("--train_file", type=str, required=True,
                    help="Path to the training CSV file")
parser.add_argument("--val_file", type=str, required=True,
                    help="Path to the validation CSV file")
parser.add_argument("--test_file", type=str, required=True,
                    help="Path to the training CSV file")
parser.add_argument("--output_dir", type=str, default="./output_final/",
                    help="Directory to save model checkpoints")
parser.add_argument("--logging_dir", type=str, default="./logs",
                    help="Directory to save training logs")
parser.add_argument("--dataset", type=str, default="pepbench",
                    help="Directory to save model checkpoints")
args = parser.parse_args()


# ----------------------
# Utilities
# ----------------------

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.clamp(torch.exp(x - m).sum(dim=dim), min=1e-12))


class _CosineMaskSchedule:
    """Cosine schedule producing masking probability p(t) in [p0, pT]."""

    def __init__(self, T: int = 100, p0: float = 0.001, pT: float = 0.999):
        self.T = int(T)
        self.p0 = float(p0)
        self.pT = float(pT)

    def p_mask(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0..T]
        # cosine from 0->1
        tt = t.float() / float(self.T)
        c = 0.5 * (1.0 - torch.cos(np.pi * tt))
        return self.p0 + (self.pT - self.p0) * c


@dataclass
class DiffusionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    protein_len: torch.Tensor  # [B]
    peptide_len: torch.Tensor  # [B]


class ProteinDataset(Dataset):
    """Produces complex sequence: protein + <mask>*K; labels: only peptide tokens supervised."""

    def __init__(self, file, tokenizer, max_length: int = 552):
        data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.proteins = data["Receptor Sequence"].tolist()
        self.peptides = data["Binder"].tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        peptide_seq = self.peptides[idx]

        # Diffusion training still uses the same I/O tensors as baseline.
        masked_peptide = "<mask>" * len(peptide_seq)
        complex_seq = protein_seq + masked_peptide

        complex_input = self.tokenizer(
            complex_seq,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        input_ids = complex_input["input_ids"].squeeze(0)
        attention_mask = complex_input["attention_mask"].squeeze(0)

        label_seq = protein_seq + peptide_seq
        labels = self.tokenizer(
            label_seq,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"].squeeze(0)

        # Only positions corresponding to masks are supervised
        labels = torch.where(input_ids == self.tokenizer.mask_token_id, labels, torch.full_like(labels, -100))
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class MotifAnchorPointerGenerator(torch.nn.Module):
    """Sequence-only conditional discrete diffusion implemented via iterative mask denoising.

    Trainer compatibility: forward returns {"loss": loss, "logits": logits}.
    Internally we re-use an ESM MLM backbone as denoiser with random mask corruption and an
    auxiliary contrastive alignment loss between protein and peptide representations.
    """

    def __init__(self, base_model: EsmForMaskedLM, tokenizer: AutoTokenizer, T: int = 100,
                 contrast_weight: float = 0.2, num_negatives: int = 4):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.T = int(T)
        self.schedule = _CosineMaskSchedule(T=self.T)
        self.contrast_weight = float(contrast_weight)
        self.num_negatives = int(num_negatives)

        # Cached indices for efficient aa-only sampling
        aa_ids = []
        for aa in AA20:
            ids = self.tokenizer.encode(aa, add_special_tokens=False)
            if len(ids) == 1:
                aa_ids.append(ids[0])
        if len(aa_ids) < 20:
            # fall back: use tokenizer vocab search for single-char AAs
            aa_ids = []
            vocab = self.tokenizer.get_vocab()
            for aa in AA20:
                if aa in vocab:
                    aa_ids.append(vocab[aa])
        self.register_buffer("aa_token_ids", torch.tensor(aa_ids, dtype=torch.long), persistent=False)

    def _infer_protein_peptide_regions(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # peptide region defined as positions where labels != -100
        peptide_mask = labels.ne(-100)
        # protein region: attention_mask==1 but not peptide
        # we avoid dependence on attention_mask here, because labels already marks peptide tokens
        protein_mask = ~peptide_mask
        return protein_mask, peptide_mask

    def _make_noised_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates P_t by randomly masking a subset of peptide tokens according to p(t)."""
        device = input_ids.device
        B, L = input_ids.shape
        protein_mask, peptide_mask = self._infer_protein_peptide_regions(input_ids, labels)

        # sample t per example
        t = torch.randint(low=1, high=self.T + 1, size=(B,), device=device)
        p = self.schedule.p_mask(t).clamp(0.0, 1.0)  # [B]

        # mask decisions for all positions (only applied on peptide region)
        # vectorized Bernoulli per example
        rand = torch.rand((B, L), device=device)
        p_row = p[:, None]
        to_mask = (rand < p_row) & peptide_mask

        noised = input_ids.clone()
        noised[to_mask] = self.tokenizer.mask_token_id

        # Training target remains original labels (already -100 outside peptide).
        return noised, t

    def _pool_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden [B,L,H], mask [B,L] bool
        # stable mean pooling, avoid division by 0
        mask_f = mask.to(hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask_f.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        return pooled

    def _contrastive_loss(self, hidden: torch.Tensor, protein_mask: torch.Tensor, peptide_mask: torch.Tensor) -> torch.Tensor:
        """InfoNCE between protein pooled embedding and peptide pooled embedding.

        Negatives: other peptides in batch (up to num_negatives) via in-batch sampling.
        """
        # hidden from MLM backbone (last hidden state)
        z_s = torch.nn.functional.normalize(self._pool_hidden(hidden, protein_mask), dim=-1)
        z_p = torch.nn.functional.normalize(self._pool_hidden(hidden, peptide_mask), dim=-1)

        B = hidden.size(0)
        if B <= 1:
            return hidden.new_zeros(())

        # logits: z_p vs z_s (positive on diagonal)
        logits = (z_p @ z_s.t()) / 0.1

        # optionally reduce to small set of negatives for memory; gather columns
        if self.num_negatives is not None and self.num_negatives > 0 and B - 1 > self.num_negatives:
            # sample negative indices per row (excluding self)
            device = hidden.device
            # create [B, B-1] candidate indices
            ar = torch.arange(B, device=device)
            all_idx = torch.arange(B, device=device).unsqueeze(0).expand(B, B)
            mask = all_idx.ne(ar.unsqueeze(1))
            candidates = all_idx[mask].view(B, B - 1)
            sel = torch.randint(0, B - 1, (B, self.num_negatives), device=device)
            neg_idx = candidates.gather(1, sel)  # [B, K]
            # columns: [pos] + negatives
            cols = torch.cat([ar[:, None], neg_idx], dim=1)  # [B, 1+K]
            small_logits = logits.gather(1, cols)
            # labels are always 0 (first column is positive)
            labels = torch.zeros(B, dtype=torch.long, device=device)
            return torch.nn.functional.cross_entropy(small_logits, labels)

        labels = torch.arange(B, device=hidden.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Trainer will pass tensors; keep signatures permissive.
        if input_ids is None or labels is None:
            raise ValueError("input_ids and labels are required")

        input_ids = input_ids
        labels = labels

        noised_ids, t = self._make_noised_inputs(input_ids, labels)
        outputs = self.model(input_ids=noised_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

        loss = outputs.loss
        hidden = outputs.hidden_states[-1]

        protein_mask, peptide_mask = self._infer_protein_peptide_regions(input_ids, labels)
        c_loss = self._contrastive_loss(hidden, protein_mask, peptide_mask)
        loss = loss + self.contrast_weight * c_loss

        return {"loss": loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate_binders(self, protein_seq: str, peptide_length: int, num_binders: int = 20,
                         top_k: int = 3, steps: Optional[int] = None, temperature: float = 1.0) -> List[str]:
        """Reverse diffusion via iterative mask-filling.

        Efficient: protein encoding recomputed each step but batch-sized across binders; avoids per-position loops.
        """
        peptide_length = int(peptide_length)
        num_binders = int(num_binders)
        top_k = int(top_k)
        steps = int(self.T if steps is None else steps)

        device = next(self.parameters()).device
        # Start from fully masked peptide region
        masked_peptide = "<mask>" * peptide_length
        seq = protein_seq + masked_peptide

        # tokenize once and repeat for num_binders
        tok = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids0 = tok["input_ids"].to(device)
        attn0 = tok["attention_mask"].to(device)
        input_ids = input_ids0.repeat(num_binders, 1)
        attention_mask = attn0.repeat(num_binders, 1)

        # determine mask positions (peptide region): current mask tokens
        # NOTE: this assumes all peptide positions start as <mask> tokens.
        mask_pos = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() != peptide_length:
            # Truncation or tokenization mismatch; best-effort: take last peptide_length tokens excluding specials
            # Avoid heavy checks in hot path.
            seq_len = input_ids.size(1)
            mask_pos = torch.arange(seq_len - peptide_length - 1, seq_len - 1, device=device)

        aa_ids = self.aa_token_ids
        if aa_ids.numel() == 0:
            raise RuntimeError("AA token ids could not be resolved for tokenizer")

        # Reverse process: iteratively unmask a subset according to decreasing p(t)
        # We deterministically choose how many to update each step, but sample tokens.
        for step in range(steps, 0, -1):
            t = torch.full((num_binders,), step, device=device, dtype=torch.long)
            p = self.schedule.p_mask(t).clamp(0.0, 1.0)  # high -> more masked
            # fraction to keep masked is p; so fraction to (re)fill is (1-p)
            fill_prob = (1.0 - p).clamp(0.0, 1.0)

            # decide which currently-masked positions to fill this step
            cur_mask = (input_ids[:, mask_pos] == self.tokenizer.mask_token_id)
            if not cur_mask.any():
                break
            rand = torch.rand_like(cur_mask.float())
            to_fill = (rand < fill_prob[:, None]) & cur_mask
            if not to_fill.any():
                continue

            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B,L,V]
            logits_pep = logits[:, mask_pos, :]

            # restrict to AA tokens, apply top-k within AA set
            aa_logits = logits_pep.index_select(-1, aa_ids)
            if temperature != 1.0:
                aa_logits = aa_logits / max(float(temperature), 1e-6)

            k = min(top_k, aa_logits.size(-1))
            topv, topi = aa_logits.topk(k, dim=-1)
            probs = torch.nn.functional.softmax(topv, dim=-1)
            # sample in vectorized manner
            sampled = torch.multinomial(probs.view(-1, k), 1).view(num_binders, -1)
            chosen_aa = topi.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            chosen_token_ids = aa_ids[chosen_aa]  # [B, pep_len]

            # apply only where to_fill
            new_pep = input_ids[:, mask_pos]
            new_pep = torch.where(to_fill, chosen_token_ids, new_pep)
            input_ids[:, mask_pos] = new_pep

        # decode peptide region
        peptides = []
        pep_tokens = input_ids[:, mask_pos]
        for i in range(num_binders):
            peptides.append(self.tokenizer.decode(pep_tokens[i], skip_special_tokens=True).replace(" ", ""))
        return peptides


# ----------------------
# Pseudo perplexity scoring (unchanged signature)
# ----------------------

def compute_pseudo_perplexity(model, tokenizer, protein_seq, binder_seq):
    sequence = protein_seq + binder_seq
    original_input = tokenizer.encode(sequence, return_tensors='pt').to(model.device)
    length_of_binder = len(binder_seq)

    masked_inputs = original_input.repeat(length_of_binder, 1)
    positions_to_mask = torch.arange(-length_of_binder - 1, -1, device=model.device)
    masked_inputs[torch.arange(length_of_binder), positions_to_mask] = tokenizer.mask_token_id

    labels = torch.full_like(masked_inputs, -100)
    labels[torch.arange(length_of_binder), positions_to_mask] = original_input[0, positions_to_mask]

    with torch.no_grad():
        outputs = model(masked_inputs, labels=labels)
        loss = outputs.loss

    avg_loss = loss.item()
    pseudo_perplexity = np.exp(avg_loss)
    return pseudo_perplexity


def generate_peptide_for_single_sequence(protein_seq, peptide_length=15, top_k=3, num_binders=4):
    peptide_length = int(peptide_length)
    top_k = int(top_k)
    num_binders = int(num_binders)

    # generation uses diffusion-capable wrapper when available (MotifAnchorPointerGenerator),
    # otherwise falls back to plain MLM single-pass.
    binders_with_ppl = []

    if isinstance(model, MotifAnchorPointerGenerator):
        peptides = model.generate_binders(protein_seq, peptide_length, num_binders=num_binders, top_k=top_k)
        for pep in peptides:
            ppl_value = compute_pseudo_perplexity(model.model, tokenizer, protein_seq, pep)
            binders_with_ppl.append([pep, ppl_value])
        return binders_with_ppl

    # Fallback (should not be the common path)
    for _ in range(num_binders):
        masked_peptide = '<mask>' * peptide_length
        input_sequence = protein_seq + masked_peptide
        inputs = tokenizer(input_sequence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        logits_at_masks = logits[0, mask_token_indices]

        top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_indices = torch.distributions.categorical.Categorical(probabilities).sample()
        predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

        generated_binder = tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')
        ppl_value = compute_pseudo_perplexity(model, tokenizer, protein_seq, generated_binder)
        binders_with_ppl.append([generated_binder, ppl_value])

    return binders_with_ppl


def generate_peptide(input_seqs, peptide_length=15, top_k=3, num_binders=4):
    if isinstance(input_seqs, str):
        binders = generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
        return pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])
    elif isinstance(input_seqs, list):
        results = []
        for seq in input_seqs:
            binders = generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
            for binder, ppl in binders:
                results.append([seq, binder, ppl])
        return pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])


# ----------------------
# 2. 加载模型和tokenizer (tokenizer MUST be consistent across train/val/test)
# ----------------------

# Use a single tokenizer instance for all stages
base_model_name = "facebook/esm2_t33_650M_UR50D"

# Train with ESM tokenizer to keep alignment; for inference, we load the same tokenizer name.
# If your checkpoints were trained with a different tokenizer (e.g., ChatterjeeLab/PepMLM-650M),
# keep it fixed by training and inference consistently within this script.

# Here we use PepMLM tokenizer (as baseline) end-to-end for strict tokenizer contract.
# We still initialize weights from ESM MLM checkpoint; tokenizers are both ESM-family and compatible.
# If the tokenizer path fails in your environment, change BOTH train/val/test together (not separately).

tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")

# ----------------------
# 3. Train: wrap denoiser with diffusion+contrastive objectives
# ----------------------

base_model = EsmForMaskedLM.from_pretrained(base_model_name)
model = MotifAnchorPointerGenerator(base_model=base_model, tokenizer=tokenizer)

lr = 0.0003
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=200,
    logging_dir=args.logging_dir,
    logging_steps=50,
    evaluation_strategy="epoch",
    # Disable Trainer's automatic checkpointing which uses safetensors by default in some envs
    save_strategy='no',
    # Do not attempt to load best model at end (we will manually save a final checkpoint)
    load_best_model_at_end=False,
    metric_for_best_model='eval_loss',
    save_total_limit=1,
    gradient_accumulation_steps=4,
    report_to="none"
)

train_dataset = ProteinDataset(args.train_file, tokenizer)
val_dataset = ProteinDataset(args.val_file, tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(AdamW(model.parameters(), lr=lr), None),
)

trainer.train()

# Manually persist a single checkpoint using PyTorch serialization to avoid safetensors shared-memory issues.
# We create a conventional HuggingFace-style folder so later loading with from_pretrained works.
os.makedirs(args.output_dir, exist_ok=True)
checkpoint_path = os.path.join(args.output_dir, "checkpoint-final")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
# Save the underlying ESM model (base_model) using torch serialization to avoid safetensors write path.
# Using safe_serialization=False forces the torch.save path and preserves shared/tied weights correctly.
try:
    # Transformers >=4.30 supports safe_serialization flag
    model.model.save_pretrained(checkpoint_path, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_path)
except TypeError:
    # Fallback for older transformers where safe_serialization flag may not exist: save state_dict with torch.save
    state_dict = model.model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_path, "pytorch_model.bin"))
    # Save config if available
    try:
        model.model.config.save_pretrained(checkpoint_path)
    except Exception:
        pass
    try:
        tokenizer.save_pretrained(checkpoint_path)
    except Exception:
        pass


# ----------------------
# 4. Load best checkpoint for generation
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expect the checkpoint folder we just wrote
# Keep the earlier contract: exactly one checkpoint folder inside output_dir -> our code writes one checkpoint-final
subdirs = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]
if len(subdirs) == 0:
    raise ValueError(f"No checkpoint folders found in {args.output_dir}")

# Prefer 'checkpoint-final' if present (we write this explicitly). Otherwise pick the latest or most recent checkpoint.
if "checkpoint-final" in subdirs:
    checkpoint_dir = "checkpoint-final"
else:
    # Prefer folder names that start with 'checkpoint' (HuggingFace Trainer style)
    ckpts = [d for d in subdirs if d.startswith("checkpoint")]
    if len(ckpts) > 0:
        # pick the checkpoint with highest step number if it exists, otherwise newest modification time
        def ckpt_key(d):
            parts = d.split("-")
            full = os.path.join(args.output_dir, d)
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
            try:
                return int(os.path.getmtime(full))
            except Exception:
                return 0
        ckpts.sort(key=ckpt_key, reverse=True)
        checkpoint_dir = ckpts[0]
    else:
        # Fall back: choose most recently modified directory
        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(args.output_dir, d)), reverse=True)
        checkpoint_dir = subdirs[0]

checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)

 
# Load base MLM then wrap with generator; state_dict contains wrapper+base weights
base_loaded = EsmForMaskedLM.from_pretrained(checkpoint_path)
model = MotifAnchorPointerGenerator(base_model=base_loaded, tokenizer=tokenizer).to(device)
model.eval()


# ----------------------
# 5. 处理test.csv (I/O contract preserved)
# ----------------------

input_df = pd.read_csv(args.test_file)

all_results = []
from tqdm import tqdm

topk = 3
print(f"\n正在处理 topk = {topk}")

# Avoid frequent GPU sync; keep memory prints minimal
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_loading_memory = torch.cuda.memory_allocated() / (1024 ** 2)
else:
    model_loading_memory = 0.0

for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
    receptor_seq = row["Receptor Sequence"]  # 受体序列
    peptide_length = row["Sequence Length"]  # 肽段长度（从原数据读取）
    num_binders = 20  # 每个受体生成20个肽段
    top_k = topk

    generated_df = generate_peptide(
        input_seqs=receptor_seq,
        peptide_length=peptide_length,
        top_k=top_k,
        num_binders=num_binders
    )

    row_info = row.drop(["Binder", "Sequence Length"]).to_dict()
    for _, gen_row in generated_df.iterrows():
        result = {
            **row_info,
            "Generated Binder": gen_row["Binder"],
            "Sequence Length": peptide_length,
            "Pseudo Perplexity": gen_row["Pseudo Perplexity"]
        }
        all_results.append(result)

output_df = pd.DataFrame(all_results)
output_df.to_csv(f"{args.output_dir}/{args.dataset}_test.csv", index=False)
print(f"生成完成，结果保存至 {args.output_dir}/{args.dataset}_test.csv")
