import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


# =====================================================================
# Architecture  (EVOLVABLE)
# =====================================================================
class CharRNN(nn.Module):
    def __init__(self, vocab, hidden_size=768, num_layers=3, dropout=0.2):
        super().__init__()
        self.vocabulary = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = self.input_size = self.output_size = len(vocab)

        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.vocab_size, padding_idx=vocab.pad
        )
        self.lstm_layer = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            dropout=dropout, batch_first=True,
        )
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        # NOTE: lengths must be on CPU for pack_padded_sequence (new PyTorch).
        x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)
        return x, lengths, hiddens

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    def sample(self, n_batch, max_length=100):
        with torch.no_grad():
            starts = torch.tensor(
                [self.vocabulary.bos] * n_batch,
                dtype=torch.long, device=self.device,
            ).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long,
                             device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)
            ]
            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1] * n_batch
            lens = torch.ones(n_batch, dtype=torch.long, device=self.device)
            end_smiles_list = [False] * n_batch

            hiddens = None
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)
                probs = [F.softmax(o, dim=-1) for o in output]
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True
                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] += 1

                starts = torch.tensor(
                    ind_tops, dtype=torch.long, device=self.device
                ).unsqueeze(1)

            new_smiles_list = [
                new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)
            ]
            return [self.tensor2string(t) for t in new_smiles_list]


# =====================================================================
# Required factory  (EVOLVABLE internals, fixed signature)
# =====================================================================
def build_model(vocab, config=None):
    """Return an (uncompiled) generative model implementing forward() + sample()."""
    config = config or {}
    return CharRNN(
        vocab,
        hidden_size=config.get("hidden", 768),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.2),
    )


# =====================================================================
# Optional custom training. Remove to fall back to the harness default
# (Adam + cross-entropy). train_loader yields (prevs, nexts, lens) and
# NEVER contains test data.
# =====================================================================
def compile_and_fit(model, train_loader, device, epochs):
    # Same optimization recipe as the reproduced MOSES baseline (Adam lr=2e-3,
    # StepLR 10/0.5, cross-entropy). Only speed is added on top and it does NOT
    # change the training math:
    #   * TF32 matmul/cudnn  -> free speedup on Ampere+/Hopper (H800).
    #   * AMP autocast       -> bf16 (preferred on Hopper, no GradScaler needed);
    #                           falls back to fp16 + GradScaler, else fp32.
    #   * torch.compile      -> opportunistic; LSTM + pack_padded_sequence may
    #                           graph-break and fall back to eager, so it is
    #                           guarded and never fatal.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    amp_dtype = None
    if use_cuda:
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    # GradScaler only matters for fp16; a no-op (enabled=False) for bf16/fp32.
    # Prefer the new torch.amp API; fall back to the deprecated one on old torch.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # Compile shares parameters with `model`, so training updates the same
    # weights the harness later samples from (in eager mode). torch.compile is
    # LAZY (it compiles on the FIRST forward), so a backend failure — e.g. the
    # inductor/Triton kernel build can't find a C compiler on this node — surfaces
    # inside the training loop, NOT at the torch.compile() call. suppress_errors
    # makes such a backend failure fall back to eager instead of crashing.
    # Production-safe: torch.compile with a fallback so a compile/backend failure
    # on some node/candidate can't crash the run (it falls back to eager). We've
    # verified compile works on the target env; suppress_errors keeps it robust
    # elsewhere. (`as _dynamo` avoids rebinding the module-level name `torch` as a
    # function local.)
    train_model = model
    if hasattr(torch, "compile"):
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            train_model = torch.compile(model)
        except Exception as _e:
            print(f"[model] torch.compile disabled, using eager: {_e}", flush=True)
            train_model = model

    train_model.train()
    for epoch in range(epochs):
        running = 0.0
        for i, (prevs, nexts, lens) in enumerate(train_loader):
            prevs = prevs.to(device)
            nexts = nexts.to(device)
            lens = lens.to(device)
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs, _, _ = train_model(prevs, lens)
                    loss = criterion(outputs.view(-1, outputs.shape[-1]),
                                     nexts.view(-1))
            else:
                outputs, _, _ = train_model(prevs, lens)
                loss = criterion(outputs.view(-1, outputs.shape[-1]),
                                 nexts.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += (loss.item() - running) / (i + 1)
        scheduler.step()
        print(f"[model] epoch {epoch + 1}/{epochs} loss={running:.4f}", flush=True)
