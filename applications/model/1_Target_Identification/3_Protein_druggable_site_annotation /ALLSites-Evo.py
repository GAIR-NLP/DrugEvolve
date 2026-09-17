
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Tuple, Optional, List

import sys
from pathlib import Path

# Keep the task-local ``src`` package importable when this file is loaded from
# the DrugEvolve candidate directory or run directly.
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.helpers import todevice
from src.models.lookahead import Lookahead
from src.models.radam import RAdam


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolutional block for feature extraction."""
    def __init__(self, in_dim: int, out_dim: int, kernel_sizes: List[int] = [3, 7, 15], dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=k//2),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in kernel_sizes])
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * len(kernel_sizes), out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, in_dim]
        Returns:
            [batch_size, seq_len, out_dim]
        """
        x_perm = x.permute(0, 2, 1)  # [batch_size, in_dim, seq_len]
        conv_outs = []
        for conv, ln in zip(self.convs, self.layer_norms):
            out = conv(x_perm).permute(0, 2, 1)  # [batch_size, seq_len, out_dim]
            out = ln(out)
            conv_outs.append(out)
        fused = torch.cat(conv_outs, dim=-1)  # [batch_size, seq_len, out_dim * 3]
        return self.fusion(fused)


class GatedCNNEncoder(nn.Module):
    """Gated CNN encoder for protein feature extraction."""
    def __init__(self, protein_dim: int, hid_dim: int = 256, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(protein_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_enc = PositionalEncoding(hid_dim, dropout=dropout)
        self.layers = nn.ModuleList([
            MultiScaleConvBlock(hid_dim, hid_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hid_dim)
    
    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein: [batch_size, seq_len, protein_dim]
        Returns:
            [batch_size, seq_len, hid_dim]
        """
        x = self.input_proj(protein)
        x = self.pos_enc(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = self.ln(x + residual)
        return x


class AttentionPooler(nn.Module):
    """Attention-based pooling for sequence features."""
    def __init__(self, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.GELU(),
            nn.Linear(hid_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hid_dim]
            mask: [batch_size, seq_len] (optional, 1 for valid positions, 0 for padding)
        Returns:
            [batch_size, hid_dim]
        """
        attn_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(1)  # [batch_size, 1, seq_len]
        pooled = torch.bmm(attn_weights, x).squeeze(1)  # [batch_size, hid_dim]
        return self.dropout(pooled)


class Predictor(nn.Module):
    """Protein-only, ligand-agnostic binding site predictor."""
    
    def __init__(self, config, device: torch.device):
        super().__init__()
        protein_dim = config['model']['protein_dim']
        local_dim = config['model']['local_dim']
        hid_dim = config['model'].get('hidden_dim', 256)
        n_layers = config['model'].get('n_layers', 3)
        dropout = config['model'].get('dropout', 0.1)
        
        # Create encoder
        self.encoder = GatedCNNEncoder(protein_dim, hid_dim, n_layers, dropout)
        self.local_proj = nn.Sequential(
            nn.Linear(local_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pooler = AttentionPooler(hid_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        self.device = device
        # Focal loss weights
        class_weight = torch.tensor([1.0, 5.0], dtype=torch.float32, device=self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    
    def make_masks(self, local_num: List[int], protein_num: List[int], 
                   local_max_len: int, protein_max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for variable-length sequences.
        """
        device = self.device

        local_num_t = torch.as_tensor(local_num, device=device, dtype=torch.long)
        protein_num_t = torch.as_tensor(protein_num, device=device, dtype=torch.long)

        local_range = torch.arange(local_max_len, device=device).unsqueeze(0)
        protein_range = torch.arange(protein_max_len, device=device).unsqueeze(0)

        local_mask = (local_range < local_num_t.unsqueeze(1)).float()
        
        if len(protein_num) == 1 and len(local_num) > 1:
            protein_mask = (protein_range < protein_num_t.unsqueeze(1)).float()
            K = len(local_num)
            protein_mask = protein_mask.expand(K, -1)
        else:
            protein_mask = (protein_range < protein_num_t.unsqueeze(1)).float()

        return local_mask, protein_mask
    
    def forward(self, local: torch.Tensor, protein: torch.Tensor, 
                local_num: List[int], protein_num: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of complete model.
        """
        local_max_len = local.shape[1]
        protein_max_len = protein.shape[1]
        local_mask, protein_mask = self.make_masks(local_num, protein_num, local_max_len, protein_max_len)
        
        # Encode protein features (runs once even if protein.shape[0]==1)
        enc_prot = self.encoder(protein)  # [1, L, H] or [batch, L, H]
        
        # Protein-grouped mode: broadcast encoded protein to all K sites
        if enc_prot.shape[0] == 1 and local.shape[0] > 1:
            K = local.shape[0]
            enc_prot = enc_prot.expand(K, -1, -1)  # [1, L, H] -> [K, L, H]
        
        # Project local features
        enc_local = self.local_proj(local)
        
        # Pool both features
        pooled_prot = self.pooler(enc_prot, protein_mask)
        pooled_local = self.pooler(enc_local, local_mask)
        
        # Concatenate and classify
        combined = torch.cat([pooled_prot, pooled_local], dim=-1)
        logits = self.classifier(combined)
        
        # Return dummy features and attention for compatibility
        return pooled_prot, torch.zeros(local.shape[0], 1, 1, 1, device=self.device), logits

    def __call__(self, data: Tuple, train: bool = True):
        """Model call for training/inference."""
        local, protein, correct_interaction, local_num, protein_num = data
        
        if train:
            sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
            del sum_features, attention  # Memory cleanup
            loss = self.loss_fn(predicted_interaction, correct_interaction)
            return loss
        else:
            sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
            del sum_features, attention  # Memory cleanup
            
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            
            return correct_labels, predicted_labels, predicted_scores


class Trainer:
    """Training class for the protein binding site predictor."""
    
    def __init__(self, model: Predictor, lr: float, weight_decay: float):
        self.model = model
        
        # Separate weight and bias parameters for different regularization
        weight_p, bias_p = [], []
        
        # Initialize parameters
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Group parameters
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        
        self.optimizer = RAdam([
            {'params': weight_p, 'weight_decay': weight_decay}, 
            {'params': bias_p, 'weight_decay': 0}
        ], lr=lr)
    
    def train(self, dataloader, device: torch.device) -> float:
        """Train the model for one epoch."""
        self.model.train()
        loss_total = 0
        
        for batch_idx, (local, protein, label, local_num, protein_num) in enumerate(dataloader):
            # Move data to device
            data_pack = todevice(local, protein, label, local_num, protein_num, device)
            
            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass and loss calculation
            loss = self.model(data_pack)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            loss_total += loss.item()
            del data_pack, loss
        
        return loss_total


class Tester:
    """Testing/evaluation class for the protein binding site predictor."""
    
    def __init__(self, model: Predictor):
        self.model = model
    
    def test(self, dataloader, device: torch.device) -> Tuple[List, List, List]:
        """Evaluate the model on test data."""
        self.model.eval()
        T, Y, S = [], [], []
        
        with torch.no_grad():
            for batch_idx, (local, protein, label, local_num, protein_num) in enumerate(dataloader):
                # Move data to device
                data_pack = todevice(local, protein, label, local_num, protein_num, device)
                
                # Get predictions
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)
                
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                
                del data_pack
        
        return T, Y, S
    
    def save_AUCs(self, AUCs: List, filename: str):
        """Save evaluation metrics to file."""
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    
    def save_model(self, model: nn.Module, filename: str):
        """Save model state dict."""
        if hasattr(model, 'module'):  # For DDP models
            torch.save(model.module.state_dict(), filename)
        else:
            torch.save(model.state_dict(), filename)
