from __future__ import annotations

"""
Bigram-anchored residual LSTM for the DrugEvolve MOSES CharRNN surface.

The fixed harness owns data loading, training invocation, decoding, and output
I/O. This file only defines the PyTorch model and an optional train loop using
train_loader batches supplied by the harness.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):
    def __init__(self, vocab, hidden_size=768, num_layers=3, dropout=0.10,
                 embedding_size=384, embedding_dropout=0.06, adapter_rank=128,
                 residual_scale=0.25, alpha_max=0.35):
        super().__init__()
        self.vocabulary = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = self.output_size = len(vocab)
        self.input_size = embedding_size
        self.alpha_max = float(alpha_max)
        self.current_alpha_max = float(alpha_max)

        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.input_size, padding_idx=vocab.pad
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.lstm_layer = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )

        self.embedding_projection = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

        self.adapter_down = nn.Linear(self.hidden_size, adapter_rank)
        self.adapter_up = nn.Linear(adapter_rank, self.output_size, bias=False)
        self.alpha_gate = nn.Linear(self.hidden_size, 1)
        nn.init.constant_(self.alpha_gate.bias, -1.0)

        initial_prior = torch.full(
            (self.vocab_size, self.vocab_size),
            -math.log(float(self.vocab_size)),
            dtype=torch.float32,
        )
        self.register_buffer("bigram_log_probs", initial_prior)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_bigram_log_probs(self, log_probs):
        if log_probs.shape != (self.vocab_size, self.vocab_size):
            raise ValueError("bigram log-probability table has an invalid shape")
        self.bigram_log_probs.copy_(log_probs.to(self.bigram_log_probs.device, dtype=torch.float32))

    def forward(self, x, lengths, hiddens=None):
        token_ids = x
        embedded = self.embedding_layer(token_ids)
        embedded = self.embedding_dropout(embedded)

        packed = rnn_utils.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hiddens = self.lstm_layer(packed, hiddens)
        recurrent, _ = rnn_utils.pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )

        residual = recurrent + self.residual_scale * self.embedding_projection(embedded)
        residual = self.output_norm(residual)
        dropped = self.output_dropout(residual)

        neural_logits = self.linear_layer(dropped)
        adapter_logits = self.adapter_up(F.gelu(self.adapter_down(dropped)))
        prior_logits = self.bigram_log_probs[token_ids]
        alpha = self.current_alpha_max * torch.sigmoid(self.alpha_gate(residual))
        logits = neural_logits + adapter_logits + alpha * prior_logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        return logits, lengths, hiddens

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    def sample(self, n_batch, max_length=100):
        # Kept for compatibility with CharRNN-style harnesses; generation is fully
        # batched and still delegates token probabilities to forward().
        was_training = self.training
        self.eval()
        with torch.no_grad():
            generated = torch.full(
                (n_batch, max_length + 2), self.vocabulary.pad,
                dtype=torch.long, device=self.device,
            )
            generated[:, 0] = self.vocabulary.bos
            starts = generated[:, :1]
            lens = torch.ones(n_batch, dtype=torch.long, device=self.device)
            lengths = torch.ones(n_batch, dtype=torch.long, device=self.device)
            finished = torch.zeros(n_batch, dtype=torch.bool, device=self.device)
            hiddens = None

            for step in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)
                probs = F.softmax(output[:, -1, :], dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(1)
                sampled = torch.where(finished, torch.full_like(sampled, self.vocabulary.pad), sampled)
                active = ~finished
                generated[:, step] = sampled
                lengths = lengths + active.long()
                finished = finished | (sampled == self.vocabulary.eos)
                starts = sampled.unsqueeze(1)

            result = [self.tensor2string(generated[i, :lengths[i].item()]) for i in range(n_batch)]
        if was_training:
            self.train()
        return result


def build_model(vocab, config=None):
    """Return an uncompiled model implementing the fixed CharRNN contract."""
    config = config or {}
    return CharRNN(
        vocab,
        hidden_size=config.get("hidden", 768),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.10),
        embedding_size=config.get("embedding", 384),
        embedding_dropout=config.get("embedding_dropout", 0.06),
        adapter_rank=config.get("adapter_rank", 128),
        residual_scale=config.get("residual_scale", 0.25),
        alpha_max=config.get("alpha_max", 0.35),
    )


def _estimate_bigram_prior(model, train_loader, device, beta=0.05):
    vocab_size = model.vocab_size
    counts = torch.zeros(vocab_size * vocab_size, dtype=torch.float32, device=device)
    pad = model.vocabulary.pad

    with torch.no_grad():
        for prevs, nexts, lens in train_loader:
            del lens
            prevs = prevs.to(device, non_blocking=True)
            nexts = nexts.to(device, non_blocking=True)
            mask = (prevs != pad) & (nexts != pad)
            pair_ids = (prevs[mask] * vocab_size + nexts[mask]).long()
            counts.add_(torch.bincount(pair_ids, minlength=vocab_size * vocab_size).to(counts.dtype))

        counts = counts.view(vocab_size, vocab_size)
        probs = counts + float(beta)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        model.set_bigram_log_probs(probs.log())


def _masked_smooth_ce(logits, targets, pad_idx, smoothing=0.003):
    log_probs = F.log_softmax(logits, dim=-1)
    safe_targets = targets.masked_fill(targets == pad_idx, 0)
    nll = -log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    if smoothing > 0.0:
        smooth = -log_probs.mean(dim=-1)
        token_loss = (1.0 - smoothing) * nll + smoothing * smooth
    else:
        token_loss = nll
    mask = (targets != pad_idx).to(token_loss.dtype)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def compile_and_fit(model, train_loader, device, epochs):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    model.to(device)
    _estimate_bigram_prior(model, train_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    total_steps = max(1, len(train_loader) * max(1, epochs))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_dtype = None
    if use_cuda:
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    model.train()
    for epoch in range(epochs):
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        batch_count = 0
        model.current_alpha_max = model.alpha_max * min(1.0, float(epoch + 1) / 1.0)
        for prevs, nexts, lens in train_loader:
            prevs = prevs.to(device, non_blocking=True)
            nexts = nexts.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs, _, _ = model(prevs, lens)
                    loss = _masked_smooth_ce(outputs, nexts, model.vocabulary.pad)
            else:
                outputs, _, _ = model(prevs, lens)
                loss = _masked_smooth_ce(outputs, nexts, model.vocabulary.pad)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum = loss_sum + loss.detach()
            batch_count += 1

        mean_loss = (loss_sum / max(1, batch_count)).item()
        print(f"[model] epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}", flush=True)
